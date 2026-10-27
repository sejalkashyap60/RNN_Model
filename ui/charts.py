import matplotlib.pyplot as plt
import numpy as np

def heatmap(values, title, x_label="Column", y_label="Row", max_side=64):
    arr=np.asarray(values)
    if arr.ndim==1: arr=arr.reshape(1,-1)
    shown=arr[:max_side,:max_side]
    fig,ax=plt.subplots(figsize=(max(5,min(12,shown.shape[1]*.16)),max(2.5,min(7,shown.shape[0]*.16))))
    image=ax.imshow(shown,aspect="auto",cmap="coolwarm",interpolation="nearest")
    ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(y_label); fig.colorbar(image,ax=ax,fraction=.025,pad=.02)
    fig.tight_layout(); return fig

def bars(values, labels, title, color="#6558df", horizontal=True):
    vals=np.asarray(values).reshape(-1); labels=list(labels)
    fig,ax=plt.subplots(figsize=(8,max(2.5,min(8,len(labels)*.34))))
    if horizontal:
        ax.barh(range(len(vals)),vals,color=color); ax.set_yticks(range(len(vals)),labels=labels); ax.invert_yaxis(); ax.set_xlabel("Value")
    else:
        ax.bar(range(len(vals)),vals,color=color); ax.set_xticks(range(len(vals)),labels=labels,rotation=60,ha="right")
    ax.set_title(title); ax.grid(axis="x" if horizontal else "y",alpha=.2); fig.tight_layout(); return fig
