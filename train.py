import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
import hydra
from omegaconf import DictConfig
import logging
from src.utils import set_seed, set_cpu_num, gumbel_sample
from tqdm import tqdm
import src.tb_writter as tb_writter
from src.model import GraphDataset, GNNPredictor
import torch.optim as optim
import numpy as np

class Trainer:
    def __init__(self,
        model: GNNPredictor,
        train_set: GraphDataset,
        valid_set: GraphDataset,
        paths: DictConfig,
        config: DictConfig,
    ) -> None:
        
        self.model = model
        self.batch_size = config.batch_size
        
        # Configure data loaders
        loader_kwargs = {
            'batch_size': self.batch_size,
            'follow_batch': ["constraint_features", "variable_features"],
            # 'num_workers': 0, # to see errors https://github.com/lucidrains/denoising-diffusion-pytorch/issues/248#issuecomment-1775323730
            # 'pin_memory': False
        }
        
        self.train_loader = DataLoader(
            train_set,
            shuffle=True,
            **loader_kwargs
        )
        
        self.valid_loader = DataLoader(
            valid_set,
            shuffle=False, 
            **loader_kwargs
        )
        
        # Configure optimizer
        output_params = {id(p) for layer in (model.vars_output_layer, model.cons_output_layer) 
                        for p in layer.parameters()}
        other_params = [p for p in model.parameters() if id(p) not in output_params]
        
        params_dict = [
            {'params': model.vars_output_layer.parameters(), 'lr': config.optim.lr_o},
            {'params': other_params, 'lr': config.optim.lr_i}
        ]
        
        optimizer_map = {
            "adam": lambda: optim.Adam(params_dict, weight_decay=config.optim.weight_decay),
            "sgd": lambda: optim.SGD(params_dict, momentum=0.9, weight_decay=1e-4)
        }
        self.optimizer = optimizer_map[config.optim.optimizer]()

        # Configure learning rate scheduler
        scheduler_map = {
            "exp": lambda: optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=config.optim.lr.anneal_factor
            ),
            "cos": lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.optim.lr.cos_T, 
                eta_min=config.optim.lr.cos_min
            ),
            "cosrestart": lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=config.optim.lr.cos_T,
                T_mult=2, eta_min=config.optim.lr.cos_min
            )
        }
        self.lr_scheduler = scheduler_map[config.optim.lr.scheduler]()
        
        # Store configuration parameters
        self.mu = config.mu.init
        self.mu_step = config.mu.step
        self.mu_step_size = config.mu.step_size
        self.mu_max = config.mu.max
        self.mu_min = config.mu.min
        self.mu_value = config.mu.value
        
        self.loss_config = config.loss_config
        self.num_samples = config.num_samples
        self.step = 0
        self.epoch = 0
        self.num_epochs = config.num_epochs
        
        # Setup model save directory
        self.model_save_dir = paths.model_save_dir
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    
    def train(self):
        best_valid_best = float('inf')
        model_path = os.path.join(self.model_save_dir, "model.pth")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Run training and validation
            train_metrics = self.run_train_epoch()
            valid_metrics = self.run_valid_epoch()
            
            # Log metrics
            self._log_metrics("Train", epoch, train_metrics)
            self._log_metrics("Valid", epoch, valid_metrics)
            
            # Save best model
            if valid_metrics["best"] <= best_valid_best:
                best_valid_best = valid_metrics["best"]
                torch.save(self.model.state_dict(), model_path)
                logging.info(f"Best model saved at epoch {epoch}.")
    
    def _log_metrics(self, phase, epoch, metrics):
        logging.info(
            f"Epoch {epoch} {phase} "
            f"loss: {metrics['loss']:0.3f} "
            f"Obj: {metrics['obj']:0.3f} " + (f"Obj_norm: {metrics['obj_norm']:0.3f} Cons_norm: {metrics['cons_norm']:0.3f} " if 'obj_norm' in metrics else '') +
            f"{phase.lower()}_cons: {metrics['cons']:0.3f}"
            + (f" Best: {metrics['best']:0.3f}" if phase == "Valid" else "")
        )

    def run_train_epoch(self):
        self.model.train()
        data_loader = self.train_loader
            
        epoch_loss, epoch_obj, epoch_cons = 0, 0, 0
        epoch_obj_norm, epoch_cons_norm, epoch_grad_norm = 0, 0, 0
        num_samples = 0
        
        for batch in tqdm(data_loader, desc="Train"):
            batch = batch.cuda()
            
            # Forward pass
            vars_o, _ = self.model.forward(batch)
            vars_o = vars_o.reshape(-1, 1)
            
            logits = unbatch(vars_o, batch=batch.variable_features_batch)
            batch = batch.to_data_list()
            
            batch_loss = torch.zeros(1, device=vars_o.device)
            batch_obj = torch.zeros(1, device=vars_o.device) 
            batch_cons = torch.zeros(1, device=vars_o.device)
            batch_cons_sum = torch.zeros(1, device=vars_o.device)
            batch_obj_norm = torch.zeros(1, device=vars_o.device)
            batch_cons_norm = torch.zeros(1, device=vars_o.device)
            batch_best = 0
            batch_best_obj = 0
            batch_mean_obj = 0
            
            # Process each graph in batch
            for i, g in enumerate(batch):
                # Sample solutions
                x = gumbel_sample(logits[i], self.num_samples, 1.0).float().reshape(self.num_samples, -1)
                x.retain_grad()
                
                # Get problem matrices
                A = g.A.cuda()
                b = g.b.cuda()
                c = g.c.cuda()

                # Calculate objectives
                p = torch.sigmoid(logits[i])
                obj = (p * c).sum()
                cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)
                cons_sum = torch.relu(A @ x.T - b).sum()
                entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum()
                
                # Calculate loss based on config
                if self.loss_config == "normalize" and torch.norm(c) > 0:
                    num_nonzero = torch.count_nonzero(cons_pos)
                    obj_norm = obj / torch.norm(c)
                    cons_norm = self.mu * (1 / torch.norm(A, dim=1) * cons_pos.squeeze()).sum() / num_nonzero if num_nonzero > 0 else 0
                    loss = cons_norm
                  
                elif self.loss_config == "sum":
                    loss = obj + self.mu * cons_pos.sum()
                elif self.loss_config == "mean":
                    loss = obj + self.mu * cons_pos.mean()
                elif self.loss_config == "nonzero_mean":
                    num_nonzero = torch.count_nonzero(cons_pos)
                    loss = obj + self.mu * cons_pos.mean() / num_nonzero if num_nonzero > 0 else obj
                
                # Validation metrics
                with torch.no_grad():
                    xx = gumbel_sample(logits[i], 500, 1.0).float().reshape(500, -1)
                    idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
                    best = (xx @ c)[idx].min().item() if len(idx) > 0 else 99999999.1337 # float('inf')
                    best_obj = (xx @ c).min().item()
                    mean_obj = (xx @ c).mean().item()
                
                batch_obj += obj
                batch_cons += cons_pos.sum()
                batch_loss += loss
                batch_obj_norm += obj_norm
                batch_cons_norm += cons_norm
                batch_cons_sum += cons_sum
                batch_best += best
                batch_best_obj += best_obj
                batch_mean_obj += mean_obj
            
            # Update running totals
            epoch_loss += batch_loss.item()
            epoch_obj += batch_obj.item()
            epoch_cons += batch_cons.item()
            epoch_obj_norm += batch_obj_norm.item()
            epoch_cons_norm += batch_cons_norm.item()
            num_samples += len(batch)
            
            # Normalize batch metrics
            batch_loss = batch_loss / len(batch)
            batch_obj = batch_obj / len(batch)
            batch_cons = batch_cons / len(batch)
            batch_cons_sum = batch_cons_sum / len(batch)
            batch_obj_norm = batch_obj_norm / len(batch)
            batch_cons_norm = batch_cons_norm / len(batch)
            batch_best = batch_best / len(batch)
            batch_best_obj = batch_best_obj / len(batch)
            batch_mean_obj = batch_mean_obj / len(batch)

            if not self.step % 100:
                if x.grad is not None:
                    x.grad.zero_()
                # obj.backward(retain_graph=True)
                # grad_obj = x.grad.clone()[0]
                # x.grad.zero_()
                cons_sum.backward(retain_graph=True)
                grad_cons_sum = x.grad.clone()[0]
                x.grad.zero_()

                print("x = ", x[0])
                print("x_grad_cons_sum = ", grad_cons_sum) 

                #print("x_grad_obj = ", grad_obj) 
            
            # Backward pass
            batch_loss.backward()
            batch_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2)
                                   for p in self.model.parameters() if p.grad is not None]), 2)


            if not self.step % 100:
                print("x = ", x[0])
                print("x_grad = ", x.grad.detach()[0]) 

            batch_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2)
                                   for p in self.model.parameters() if p.grad is not None]), 2)

            epoch_grad_norm += batch_grad_norm
            batch_grad_norm /= len(batch)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Logging
            self.step += 1
            tb_writter.set_step(self.step)
            tb_writter.add_scalar("Loss/loss", batch_loss.item(), self.step)
            tb_writter.add_scalar("Loss/obj", batch_obj.item(), self.step)
            tb_writter.add_scalar("Loss/cons", batch_cons.item(), self.step)
            tb_writter.add_scalar("Loss/cons_sum", batch_cons_sum.item(), self.step)
            tb_writter.add_scalar("Loss/obj_norm", batch_obj_norm.item(), self.step)
            tb_writter.add_scalar("Loss/cons_norm", batch_cons_norm.item(), self.step)
            tb_writter.add_scalar("Loss/entropy", entropy, self.step)
            tb_writter.add_scalar("Loss/grad_norm", batch_grad_norm, self.step)
            tb_writter.add_scalar("Params/mu", self.mu, self.step)
            tb_writter.add_scalar("Params/lr_i", self.lr_scheduler.get_last_lr()[1], self.step)
            tb_writter.add_scalar("Params/lr_o", self.lr_scheduler.get_last_lr()[0], self.step)
            tb_writter.add_scalar("Output/Best", batch_best, self.step)
            tb_writter.add_scalar("Output/Best_obj", batch_best_obj, self.step)
            tb_writter.add_scalar("Output/Mean_obj", batch_mean_obj, self.step)
            tb_writter.add_scalar("Output/prob", torch.sigmoid(vars_o).mean(), self.step)
            tb_writter.add_scalar("Output/logits", vars_o.mean(), self.step)
            tb_writter.add_scalar("Output/prob_min", torch.sigmoid(vars_o).min(), self.step)
            tb_writter.add_scalar("Output/logits_min", vars_o.min(), self.step)
            tb_writter.add_scalar("Output/prob_max", torch.sigmoid(vars_o).max(), self.step)
            tb_writter.add_scalar("Output/logits_max", vars_o.max(), self.step)
            tb_writter.add_histogram("Prediction/samples", x, self.step)
            tb_writter.add_histogram("Prediction/pred", torch.sigmoid(vars_o), self.step)
            tb_writter.add_histogram("Prediction/logits", vars_o, self.step)
        
        # Update learning rate and mu
        self.lr_scheduler.step()
        self.mu = self.mu + self.mu_step_size * (epoch_cons / num_samples - self.mu_value)
        self.mu = max(min(self.mu, self.mu_max), self.mu_min)
        print(f"{self.mu=}")
        
        return {
            "loss": epoch_loss / num_samples,
            "obj": epoch_obj / num_samples,
            "cons": epoch_cons / num_samples,
            "obj_norm": epoch_obj_norm / num_samples,
            "cons_norm": epoch_cons_norm / num_samples,
            "grad_norm": epoch_grad_norm / num_samples,
        }
    
    @torch.no_grad()
    def run_valid_epoch(self):
        self.model.eval()
        data_loader = self.valid_loader
            
        epoch_metrics = {
            'loss': 0, 'obj': 0, 'cons': 0, 'best': 0
        }
        num_samples = 0

        for batch in tqdm(data_loader, desc="Valid"):
            batch = batch.cuda()
            
            # Predict binary distribution
            vars_o, cons_o = self.model.forward(batch)
            vars_o = vars_o.reshape(-1, 1)
            
            logits = unbatch(vars_o, batch=batch.variable_features_batch)
            batch = batch.to_data_list()
            batch_metrics = {
                'loss': 0, 'obj': 0, 'cons': 0, 'best': 0,
                'best_obj': 0, 'mean_obj': 0
            }

            # Process each graph in batch
            for g, logit in zip(batch, logits):
                # Sample solutions
                x = gumbel_sample(logit, self.num_samples, 1.0).float().reshape(self.num_samples, -1)
                
                # Get problem matrices/vectors
                A, b, c = [tensor.cuda() for tensor in (g.A, g.b, g.c)]

                # Calculate metrics
                p = torch.sigmoid(logit)
                obj = (p * c).sum()
                cons_pos = torch.relu(A @ x.T - b).mean(dim=1, keepdim=True)
                
                # Calculate loss
                loss = obj + self.mu * cons_pos.sum()

                # Additional sampling for statistics
                xx = gumbel_sample(logit, 500, 1.0).float().reshape(500, -1)
                feasible_idx = torch.where(torch.relu(A @ xx.T - b).sum(0) == 0)[0]
                best = (xx @ c)[feasible_idx].min().item() if len(feasible_idx) > 0 else 1e3
                
                # Update batch metrics
                metrics = {
                    'obj': obj,
                    'cons': cons_pos.sum(),
                    'loss': loss,
                    'best': best,
                    'best_obj': (xx @ c).min().item(),
                    'mean_obj': (xx @ c).mean().item()
                }
                for k, v in metrics.items():
                    batch_metrics[k] += v

            # Update epoch metrics
            batch_size = len(batch)
            num_samples += batch_size
            
            for k in ['loss', 'obj', 'cons', 'best']:
                epoch_metrics[k] += batch_metrics[k]

            # Log batch metrics
            normalized_metrics = {k: v/batch_size for k,v in batch_metrics.items()}

        # Log epoch metrics
        for k, v in epoch_metrics.items():
            normalized_v = v / num_samples
            tb_writter.add_scalar(f"Valid/{k}", normalized_v, self.epoch)
            epoch_metrics[k] = normalized_v

        return epoch_metrics

@hydra.main(version_base=None, config_path="config", config_name="train")
def train(config: DictConfig):
    """
    Train the model.
    """
    
    # Initialize settings
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)
    torch.cuda.set_device(config.cuda)
    tb_writter.set_logger(config.paths.tensorboard_dir)
    
    # Get all sample files and split into train/valid
    sample_files = [os.path.join(config.paths.data_samples_dir, f) 
                   for f in os.listdir(config.paths.data_samples_dir)]
    split_idx = int(0.80 * len(sample_files))
    train_files, valid_files = sample_files[:split_idx], sample_files[split_idx:]
    
    print(f"{len(sample_files)=}")
    
    # Create datasets
    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)
    
    # Initialize model and move to GPU
    device = torch.device(f'cuda:{config.cuda}')
    model = GNNPredictor(config.model).to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        train_set=train_data,
        valid_set=valid_data,
        paths=config.paths,
        config=config.trainer,
    )
    
    trainer.train()

if __name__ == "__main__":
    train()
    
