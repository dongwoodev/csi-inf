import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin, scale):
        """
        ### ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        (https://www.kaggle.com/code/nanguyen/arcface-loss)
        
        Args : 
            `self.num_classes` : 분류해야할 클래스 수
            `self.embedding_size` : 임베딩 벡터의 크기
            `self.margin` : 클래스 간 각도를 의미함.
            `self.scale` : 임베딩의 스케일 조정
            `self.W` : 가중치 행렬로 각 클래스에 대한 임베딩 정의
               - 행렬 크기 (num_classes, embediing_size)
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size # 임베딩 벡터 크기
        self.margin = margin 
        self.scale = scale

        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))

        nn.init.xavier_normal_(self.W) # 정규 초기화

    def forward(self, embeddings, labels):
        """
        Variable:
            diff : 원래 코사인 유사도와 수정된 코사인 유사도의 차이
            logits : 코사인 유사도에 수정된 차이 적용하고, 최종 logits 값을 계산하기 위해 스케일 조정

        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar        
            - 손실 계산 후 반환
        """
        cosine = self.get_cosine(embeddings)
        mask = self.get_target_mask(labels).expand_as(cosine)
        # cosine(64,64)과 mask(64,3)의 사이즈 불일치
        cosine_of_target_classes = cosine[mask == 1] # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        ) # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)
        logits = self.scale_logits(logits) # (None, n_classes)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def get_cosine(self, embeddings):
        """
        ### get_cosine
        임베딩 벡터와 가중치 행렬의 선형 변환으로 코사인 값을 구함
        
        Args:
            embeddings : (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine
    
    def get_target_mask(self, labels):
        """
        ### get_target_mask
        주어진 레이블에 대해 One-Hot Encoding 수행
        
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot
    
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        ### modify_cosine_of_target_classes
        주어진 코사인 값에 대해 각도 마진 적용.
        - torch.acos : 코사인 값을 계산 
        - 리턴 시, 마진을 추가하여 코사인 값으로 변환

        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)
    
    def scale_logits(self, logits):
        """
        ### scale-logits
        로짓값을 스케일 매개변수로 조정 할 수 있음.

        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale
    

"""
        if loss_type == 'arcface':
            self.loss_function = ArcFaceLoss(
                num_classes=10, 
                embedding_size=embedding_size,
                margin=0.3, 
                scale=30.0
            )
"""