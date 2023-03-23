# Evaluation

## Metrics

### Average Precision (AP)

$$
\text{AP}=\frac{1}{101}\underset{t \in \mathcal{T}}{\sum}\underset{r\in\mathcal{R}}{\sum}\text{p}_{\text{interp}}(r)
$$

### Average Translation Error (ATE)

$$
\text{ATE}=||\text{t}_{\text{dt}}-\text{t}_{\text{gt}}||_2\quad\text{where}\quad\text{t}_{\text{dt}}\in\mathbb{R}^3,\text{t}_{\text{gt}}\in\mathbb{R}^3
$$

### Average Scale Error (ASE)

$$
\text{ASE}=1-\underset{d\in\mathcal{D}}{\prod}\frac{\min(d_{\text{dt}},d_{\text{gt}})}{\max(d_{\text{dt}},d_\text{gt})}
$$

### Average Orientation Error (AOE)

$$
\text{AOE}=|\theta_{\text{dt}}-\theta_{\text{gt}}|\\\theta_{\text{dt}}\in[0,\pi),\theta_{\text{gt}}\in[0,\pi)
$$

### Composite Detection Score (CDS)

$$
\text{CDS}=\text{mAP}\cdot\underset{x\in\mathcal{X}}{\sum}1-x\\\mathcal{X}=\{\text{mATE}_{\text{unit}},\text{mASE}_{\text{unit}},\text{mAOE}_{\text{unit}}\}
$$