import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from alpha_3_multiclass_model import Alpha_3_multiclass_model
from utils_tests import (
    seed_everything,
    preprocess_train_test_dataset_for_alpha_3,
    BertEmbeddingDataset,
)


class Alpha_3_multiclass_trainer_tests:
    def __init__(
        self,
        number_of_epochs: int,
        train_path: str,
        test_path: str,
        seed: int,
        n_qubits: int,
        q_delta: float,
        batch_size: int,
        lr: float,
        weight_decay: float,
        step_lr: int,
        gamma: float,
    ):

        self.number_of_epochs = number_of_epochs
        self.train_path = train_path
        self.test_path = test_path
        self.seed = seed
        self.n_qubits = n_qubits
        self.q_delta = q_delta
        self.batch_size = batch_size

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_lr = step_lr
        self.gamma = gamma

        # seed everything
        seed_everything(self.seed)

        (
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
        ) = preprocess_train_test_dataset_for_alpha_3(
            self.train_path, self.test_path
        )

        self.n_classes = self.Y_train.apply(tuple).nunique()

        print("In the dataset there is:", self.n_classes, "classes")

        # initialise datasets and optimizers as in PyTorch

        self.train_dataset = BertEmbeddingDataset(self.X_train, self.Y_train)
        self.training_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Shuffle is set to False for the test dataset because in the predict function we need to keep the order of the predictions
        self.test_dataset = BertEmbeddingDataset(self.X_test, self.Y_test)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        """
        self.dummy_dataset = BertEmbeddingDataset(self.X_dummy, self.Y_dummy)
        self.dummy_dataloader = DataLoader(
            self.dummy_dataset, batch_size=self.batch_size, shuffle=False
        )
        """
        # initialise the device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # initialise model
        self.model = Alpha_3_multiclass_model(
            self.n_qubits, self.q_delta, self.n_classes, self.device
        )

        # initialise loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_lr, gamma=self.gamma
        )

        # self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self):
        training_loss_list = []
        training_acc_list = []

        test_loss_list = []
        test_acc_list = []

        best_test_acc = 0.0

        for epoch in range(self.number_of_epochs):
            print("Epoch: {}".format(epoch))
            running_loss = 0.0
            running_corrects = 0

            self.model.train()
            # with torch.enable_grad():
            # for circuits, embeddings, labels in train_dataloader:
            for inputs, labels in self.training_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # print('preds: ', preds)
                # print('labels: ', torch.max(labels, 1)[1])

                self.optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_

                batch_corrects = torch.sum(
                    preds == torch.max(labels, 1)[1]
                ).item()
                running_corrects += batch_corrects

            # Print epoch results
            train_loss = running_loss / len(self.training_dataloader.dataset)
            train_acc = running_corrects / len(
                self.training_dataloader.dataset
            )

            training_loss_list.append(train_loss)
            training_acc_list.append(train_acc)

            running_loss = 0.0
            running_corrects = 0

            self.model.eval()

            with torch.no_grad():
                for inputs, labels in self.test_dataloader:
                    batch_size_ = len(inputs)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(
                        preds == torch.max(labels, 1)[1]
                    ).item()
                    running_corrects += batch_corrects

            test_loss = running_loss / len(self.test_dataloader.dataset)
            test_acc = running_corrects / len(self.test_dataloader.dataset)

            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = self.model.state_dict()

            self.lr_scheduler.step()

            print("Train loss: {}".format(train_loss))
            print("Test loss: {}".format(test_loss))
            print("Train acc: {}".format(train_acc))
            print("Test acc: {}".format(test_acc))

            print("-" * 20)

        return (
            training_loss_list,
            training_acc_list,
            test_loss_list,
            test_acc_list,
            best_test_acc,
            best_model,
        )

    def predict(self):
        prediction_list = torch.tensor([]).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                prediction_list = torch.cat(
                    (prediction_list, torch.round(torch.flatten(preds)))
                )

        return prediction_list.detach().cpu().numpy()

    def compute_dummy_logs(self, best_model):
        running_loss = 0.0
        running_corrects = 0

        # Load the best model found during training
        self.model.load_state_dict(best_model)
        self.model.etest()

        with torch.no_grad():
            for inputs, labels in self.dummy_dataloader:
                batch_size_ = len(inputs)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(
                    preds == torch.max(labels, 1)[1]
                ).item()
                running_corrects += batch_corrects

        dummy_loss = running_loss / len(self.dummy_dataloader.dataset)
        dummy_acc = running_corrects / len(self.dummy_dataloader.dataset)

        print("Run dummy results:")
        print("dummy loss: {}".format(dummy_loss))
        print("dummy acc: {}".format(dummy_acc))

        return dummy_loss, dummy_acc
