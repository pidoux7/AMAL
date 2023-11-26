import optuna
import torch.nn as nn


def objective(trial):
    from tp7 import Model, run, NUM_CLASSES, INPUT_DIM
    iterations = 200
    dims = [100, 100, 100]

    norm_type = trial.suggest_categorical('normalization', ["identity", "batchnorm", "layernorm"])
    normalization = norm_type

    dropouts = [trial.suggest_loguniform('dropout_p%d' % ix, 1e-2, 0.5) for ix in range(len(dims))]

    l2 = trial.suggest_uniform('l2', 0, 1)
    l1 = trial.suggest_uniform('l1', 0, 1)

    model = Model(INPUT_DIM, NUM_CLASSES, dims, dropouts, normalization)
    return run(iterations, model, l1, l2)

study = optuna.create_study()
study.optimize(objective, n_trials=20)
print(study.best_params)









'''

class Lit3Layer(pl.LightningModule):
    def __init__(self,dim_in,l,dim_out,learning_rate=1e-3):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim_in,l),nn.ReLU(),nn.Linear(l,l),nn.ReLU(),nn.Linear(l,dim_out))
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "exemple-lightning"
        self.valid_outputs = []
        self.training_outputs = []

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x),on_step=False,on_epoch=True)
        self.valid_outputs.append({"loss":loss,"accuracy":acc,"nb":len(x)})
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        return logs

    def log_x_end(self,outputs,phase):
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/{phase}":total_loss,f"acc/{phase}":total_acc})
        #self.logger.experiment.add_scalar(f'loss/{phase}',total_loss,self.current_epoch)
        #self.logger.experiment.add_scalar(f'acc/{phase}',total_acc,self.current_epoch)

    def on_training_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        self.log_x_end(self.training_outputs,'train')
        self.training_outputs.clear()
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX
    def on_validation_epoch_end(self):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        self.log_x_end(self.valid_outputs,'valid')
        self.valid_outputs.clear()

    def on_test_epoch_end(self):
        pass

model = Lit3Layer(dim_in,100,dim_out)
logger = TensorBoardLogger('tb_logs', name=model.name)
trainer = pl.Trainer( max_epochs=10, logger=logger)
trainer.fit(model,train_loader,test_loader)
trainer.test(model,test_dataloaders=test_loader)

'''