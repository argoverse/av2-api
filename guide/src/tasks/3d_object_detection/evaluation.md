# Evaluation

## Metrics

### Average Precision (AP):

$$
\begin{align}
    \text{AP}=\frac{1}{100}\underset{t \in \mathcal{T}}{\sum}\underset{r\in\mathcal{R}}{\sum}\text{p}_{\text{interp}}(r)
\end{align}
$$

### Average Translation Error (ATE):

$$
\begin{align}
    \text{ATE} = \lVert t_{\text{dt}}-t_{\text{gt}} \rVert_2 \quad \text{where} \quad t_{\text{dt}}\in\mathbb{R}^3,t_{\text{gt}}\in\mathbb{R}^3
\end{align}
$$

### Average Scale Error (ASE):

$$
\begin{align}
    \text{ASE} = 1 - \underset{d\in\mathcal{D}}{\prod}\frac{\min(d_{\text{dt}},d_{\text{gt}})}{\max(d_{\text{dt}},d_\text{gt})}
\end{align}
$$

### Average Orientation Error (AOE):

$$
\begin{align}
    \text{AOE} = |\theta_{\text{dt}}-\theta_{\text{gt}}| \quad \text{where} \quad \theta_{\text{dt}}\in[0,\pi) \text{ and } \theta_{\text{gt}}\in[0,\pi)
\end{align}
$$

### Composite Detection Score (CDS):

$$
\begin{align}
    \text{CDS}&= \text{mAP} \cdot \underset{x\in\mathcal{X}}{\sum}{ 1-x } \\\mathcal{X}&=\{\text{mATE}_{\text{unit}},\text{mASE}_{\text{unit}},\text{mAOE}_{\text{unit}}\}
\end{align}
$$
