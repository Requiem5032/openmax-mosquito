import matplotlib.pyplot as plt


def draw_learning_curve(history, validation=True):
    if validation:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(loss, color='teal', label='loss')
        plt.plot(val_loss, color='orange', label='val_loss')
        plt.legend(loc='upper left')
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(acc, color='teal', label='accuracy')
        plt.plot(val_acc, color='orange', label='val_accuracy')
        plt.legend(loc='upper left')
        plt.title('Training and Validation Accuracy')

        plt.show()

    else:
        loss = history.history['loss']
        acc = history.history['accuracy']

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(loss, color='teal', label='loss')
        plt.legend(loc='upper left')
        plt.title('Training Loss')

        plt.subplot(1, 2, 2)
        plt.plot(acc, color='teal', label='accuracy')
        plt.legend(loc='upper left')
        plt.title('Training Accuracy')

        plt.show()


def draw_confusion_matrix(cm, labels):
    class_num = len(labels)
    tick_marks = range(class_num)
    thresh = cm.min() + (cm.max()-cm.min()) / 2

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.matshow(cm, cmap=plt.cm.gray_r)
    fig.colorbar(im)

    for i in range(class_num):
        for j in range(class_num):
            fg = None
            if cm[i, j] < thresh:
                fg = 'black'
            else:
                fg = 'white'
            ax.text(
                x=j,
                y=i,
                s=cm[i, j],
                ha='center',
                va='center',
                ma='center',
                color=fg,
                size='x-large',
            )

    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    plt.setp(
        ax.set_xticklabels(labels),
        ha='right',
        va='top',
        ma='center',
        rotation=45,
        rotation_mode='default',
    )
    plt.setp(
        ax.set_yticklabels(labels),
        ha='right',
        va='center',
        ma='center',
        rotation='horizontal',
        rotation_mode='default',
    )
