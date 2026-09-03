"""
architecture.py
Modèle GPT minimal, inspiré de nanoGPT (Karpathy), pour Sabrina: Codex.

Pensé pour rester lisible et modifiable : pas de dépendance à transformers,
tout est écrit en PyTorch pur pour que tu comprennes/contrôles chaque brique.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import SabrinaConfig


class CausalSelfAttention(nn.Module):
    """Auto-attention masquée (chaque token ne voit que les tokens précédents)."""

    def __init__(self, config: SabrinaConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Une seule projection linéaire pour Q, K, V combinés (plus efficace)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Masque causal : un token ne peut pas "voir" le futur
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, séquence, embedding_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Réorganise pour le multi-head : (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention scaled dot-product
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    """Feed-forward classique : projection vers un espace plus grand, puis retour."""

    def __init__(self, config: SabrinaConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloc Transformer : attention + feed-forward, avec normalisation et résiduelles."""

    def __init__(self, config: SabrinaConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SabrinaCodex(nn.Module):
    """Le modèle complet : embeddings + N blocs Transformer + tête de sortie."""

    def __init__(self, config: SabrinaConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying : la tête de sortie partage la même matrice que l'embedding
        # d'entrée (pratique standard GPT-2/nanoGPT). Économise vocab_size × n_embd
        # paramètres — dans notre cas, ~4,1M sur 9,02M au total — sans perte de
        # qualité connue, et libère ce budget pour les vraies couches Transformer.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Sabrina: Codex] Modèle initialisé — {n_params/1e6:.2f}M paramètres")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Séquence plus longue que block_size"

        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        eos_token_id: int = None,
        repetition_penalty: float = 1.0,
    ):
        """Génère du texte token par token à partir d'un contexte de départ.

        eos_token_id : si fourni, arrête la génération dès que ce token est produit
            (au lieu de toujours générer max_new_tokens) — évite de continuer à
            générer après un point d'arrêt naturel (ex: <|endofcode|>).
        repetition_penalty : > 1.0 pénalise les tokens déjà présents dans la séquence
            générée, pour réduire les boucles de répétition. 1.0 = désactivé.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for batch_idx in range(idx.shape[0]):
                    for token_id in set(idx[batch_idx].tolist()):
                        score = logits[batch_idx, token_id]
                        logits[batch_idx, token_id] = (
                            score / repetition_penalty if score > 0 else score * repetition_penalty
                        )

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and idx.shape[0] == 1 and next_token.item() == eos_token_id:
                break

        return idx