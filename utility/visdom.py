import numpy as np
from visdom import Visdom

def visdom_loss():
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([int(count_loss) / count_image]),
        win=visdom_win_loss, #win要保持一致
        name='validation_loss',
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='loss %',
        )
    )

def visdom_acc():
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([int(count_acc) / count_image]),
        win=visdom_win_acc, #win要保持一致
        name='validation_acc',
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='acc %',
        )
    )

def visdom_se():
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([se]),
        win=visdom_win_se_and_sp_validation,
        name='validation_se',
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='validation se and sp',
        )
    )

def visdom_sp():
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([sp]),
        win=visdom_win_se_and_sp_validation,
        name='validation_sp',
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='validation se and sp',
        )
    )

def visdom_roc_auc():
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([roc_auc]),
        win=visdom_win_roc_auc,
        name='validation_roc_auc',
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='roc_auc',
        )
    )
