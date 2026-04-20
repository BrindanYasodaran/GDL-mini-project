import random
random.seed(0)
import numpy as np
import torch
from torch_geometric.data import Data
from easydict import EasyDict




class RadiusProblemGraphs(object):
    def __init__(self, args:EasyDict, add_crosses, num_classes):
        self.args = args
        self.n = args.n
        self.depth = args.depth
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples
        self.need_one_hot = args.need_one_hot
        self.add_crosses = add_crosses
        self.num_classes = num_classes
        self.repeat = args.repeat

    def one_hot_encode(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        One-hot encode indices using torch.nn.functional.one_hot
        without `import torch.nn.functional as F`.
        """
        oh = torch.nn.functional.one_hot(indices, num_classes=num_classes)
        return oh.float()
    
    def generate_sample(self):
        raise NotImplementedError
    
    def generate_dataset(self,num_samples):
        """
        Generate a dataset of lollipop transfer graphs.
        Returns:
        - list[Data]: List of Torch geometric data structures.
        """
        nodes = self.n
        if nodes <= 1: raise ValueError("Minimum of two nodes required")
        dataset = []
        samples_per_class = num_samples // self.num_classes
        for i in range(num_samples):
            label = i // samples_per_class
            target_class = np.zeros(self.num_classes)
            target_class[label] = 1.0
            dataset.append(self.generate_sample(nodes,label))
        return dataset
    
    def generate_data(self):
        """
        Wrapper for generating the train/test/validation splits.
        Returns three lists of Data objects: (train, test, val).
        """
        train = self.generate_dataset(self.num_train_samples)
        test = self.generate_dataset(self.num_test_samples)
        val = self.generate_dataset(self.num_test_samples)
        return (train, test, val)


class OneRadiusProblemStarGraph(RadiusProblemGraphs):
    def __init__(self, args: EasyDict, num_classes=10):
        self.args = args
        self.n = args.n
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples
        self.num_classes = num_classes
        self.need_one_hot = True
        self.K = 1
        args.num_classes = num_classes

    def one_hot_encode(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

    def generate_sample(self, n, label=None):
        """
        Generate a single star-graph with one center node.
        """
        total_nodes = n + 1
        labels_A = torch.randint(0, self.num_classes, (n,), dtype=torch.long)
        
        random_ids_A = random.sample(range(1, n + 1), n)
        center_id = random.choice(random_ids_A)
        
        # Build edges
        edge_index = []
        for i in range(1, total_nodes):
            edge_index.append([0, i])
            edge_index.append([i, 0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Build features
        features_A = [[random_ids_A[i-1], labels_A[i-1].item()] for i in range(1, total_nodes)]
        
        center_node = 0
        if self.need_one_hot:
            features_A_oh = [
                torch.cat((
                    self.one_hot_encode(torch.tensor(f[0], dtype=torch.long), n+1),
                    self.one_hot_encode(torch.tensor(f[1], dtype=torch.long), self.num_classes + 1)
                ))
                for f in features_A
            ]
            features_A_oh = torch.stack(features_A_oh, dim=0)
            feature_v_oh = torch.cat((
                self.one_hot_encode(torch.tensor(center_id, dtype=torch.long), n+1),
                self.one_hot_encode(torch.tensor(self.num_classes, dtype=torch.long), self.num_classes + 1)
            ))
            x = torch.cat([feature_v_oh.unsqueeze(0), features_A_oh], dim=0)
            center_label = labels_A[random_ids_A.index(center_id)]
        else:
            feature_v = [center_id, self.num_classes]
            x = torch.tensor(features_A + [feature_v], dtype=torch.long)
            center_label = labels_A[random_ids_A.index(center_id)]

        # Create mask and target
        mask = torch.zeros(total_nodes, dtype=torch.bool)
        mask[center_node] = True
        y = torch.tensor([center_label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y,
                train_mask=mask, val_mask=mask, test_mask=mask)
        
    def generate_dataset(self, num_samples):
        """Generate dataset."""
        nodes = self.n
        if nodes <= 1:
            raise ValueError("Minimum of two nodes required")
        dataset = []
        for _ in range(num_samples):
            dataset.append(self.generate_sample(nodes))
        return dataset
    
    def generate_data(self):
        """Generate train/test/validation splits."""
        train = self.generate_dataset(self.num_train_samples)
        test = self.generate_dataset(self.num_test_samples)
        val = self.generate_dataset(self.num_test_samples)
        return (train, test, val)


class TwoRadiusProblemStarGraph(RadiusProblemGraphs):
    """
    Base class for two-radius problem (connected variant with K central nodes).
    """
    def __init__(self, args: EasyDict, num_classes=10):
        self.args = args
        self.n = args.n
        self.K = getattr(args, 'K', 1)
        self.num_train_samples = args.num_train_samples
        self.num_test_samples = args.num_test_samples
        self.num_classes = num_classes
        self.need_one_hot = True
        args.num_classes = num_classes

    def one_hot_encode(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

    def _build_edge_index(self, n):
        """
        Build edge index for connected graph.
        Connect A nodes to all central nodes, and central nodes to all B nodes.
        """
        edge_index = []
        for i in range(n):
            for j in range(self.K):
                center_idx = n + j
                edge_index.append([i, center_idx])
                edge_index.append([center_idx, i])
        for j in range(self.K):
            for i in range(n):
                center_idx = n + j
                b_idx = n + self.K + i
                edge_index.append([center_idx, b_idx])
                edge_index.append([b_idx, center_idx])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def generate_sample(self, n, label=None):
        """Generate single graph sample."""
        num_nodes = 2 * n + self.K
        
        edge_index = self._build_edge_index(n)

        A_ids = torch.tensor(random.sample(range(n), n), dtype=torch.long)
        C_ids = torch.arange(n, n + self.K, dtype=torch.long)
        B_ids = A_ids[torch.randperm(n)]

        A_labels = torch.randint(0, self.num_classes, (n,), dtype=torch.long)
        C_labels = torch.full((self.K,), 0, dtype=torch.long)
        B_labels = torch.full((n,), self.num_classes - 1, dtype=torch.long)

        A_ids_oh = self.one_hot_encode(A_ids, n + self.K)
        C_ids_oh = self.one_hot_encode(C_ids, n + self.K)
        B_ids_oh = self.one_hot_encode(B_ids, n + self.K)

        A_labels_oh = self.one_hot_encode(A_labels, self.num_classes)
        C_labels_oh = self.one_hot_encode(C_labels, self.num_classes)
        B_labels_oh = self.one_hot_encode(B_labels, self.num_classes)

        features_A = torch.cat((A_ids_oh, A_labels_oh), dim=1)
        features_C = torch.cat((C_ids_oh, C_labels_oh), dim=1)
        features_B = torch.cat((B_ids_oh, B_labels_oh), dim=1)
        x = torch.cat((features_A, features_C, features_B), dim=0)

        id_to_label = {aid.item(): lab.item() for aid, lab in zip(A_ids, A_labels)}
        B_true_labels = [id_to_label[bid.item()] for bid in B_ids]
        y = torch.tensor(B_true_labels, dtype=torch.long)

        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[n + self.K:] = True
        
        sources = B_ids.squeeze()
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y,
            train_mask=mask, 
            val_mask=mask, 
            test_mask=mask,
            sources=sources,
            n=self.n 
        )
        return data

    def generate_dataset(self, num_samples):
        """Generate dataset."""
        nodes = self.n
        if nodes <= 1:
            raise ValueError("Minimum of two nodes required")
        dataset = []
        for _ in range(num_samples):
            dataset.append(self.generate_sample(nodes))
        return dataset

    def generate_data(self):
        """Generate train/test/validation splits."""
        train = self.generate_dataset(self.num_train_samples)
        test = self.generate_dataset(self.num_test_samples)
        val = self.generate_dataset(self.num_test_samples)
        return (train, test, val)


class TwoRadiusProblemDisconnectedGraph(TwoRadiusProblemStarGraph):
    """
    Disconnected variant: inherits from TwoRadiusProblemStarGraph.
    Only overrides edge building and removes central nodes.
    """
    def __init__(self, args: EasyDict, num_classes=10):
        super().__init__(args, num_classes)
        self.K = 0  # No central nodes

    def _build_edge_index(self, n):
        """Override to create empty edge index."""
        return torch.empty((2, 0), dtype=torch.long)

    def generate_sample(self, n, label=None):
        """Generate disconnected graph sample."""
        num_nodes = 2 * n

        edge_index = self._build_edge_index(n)

        A_ids = torch.tensor(random.sample(range(n), n), dtype=torch.long)
        B_ids = A_ids[torch.randperm(n)]

        A_labels = torch.randint(0, self.num_classes, (n,), dtype=torch.long)
        B_labels = torch.full((n,), self.num_classes - 1, dtype=torch.long)

        A_ids_oh = self.one_hot_encode(A_ids, n)
        B_ids_oh = self.one_hot_encode(B_ids, n)

        A_labels_oh = self.one_hot_encode(A_labels, self.num_classes)
        B_labels_oh = self.one_hot_encode(B_labels, self.num_classes)

        features_A = torch.cat((A_ids_oh, A_labels_oh), dim=1)
        features_B = torch.cat((B_ids_oh, B_labels_oh), dim=1)
        x = torch.cat((features_A, features_B), dim=0)

        id_to_label = {aid.item(): lab.item() for aid, lab in zip(A_ids, A_labels)}
        B_true_labels = [id_to_label[bid.item()] for bid in B_ids]
        y = torch.tensor(B_true_labels, dtype=torch.long)

        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[n:] = True

        sources = B_ids.squeeze()
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=mask,
            val_mask=mask,
            test_mask=mask,
            sources=sources,
            n=self.n
        )
        return data
        
        
        
        
        
          
        
        
        