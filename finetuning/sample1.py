import train
import matplotlib.pyplot as plt

loss_list = []
val_loss_list = []
val_acc_list = []
epoch = 100


def main():
    print("a")
    # t = train.Train()
    # t.set_train_config()
    #
    # for e in range(epoch):
    #     train_loss = t.train_start()
    #     val_loss, val_acc = t.val_start()
    #     loss_list.append(train_loss)
    #     val_loss_list.append(val_loss)
    #     val_acc_list.append(val_acc_list)
    #
    # plt.figure()
    # plt.plot(range(epoch), loss_list, 'r-', label='train_loss')
    # plt.plot(range(epoch), val_loss_list, 'b-', label='val_loss')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(range(epoch), val_acc_list, 'g-', label='val_acc')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('acc')
    # plt.grid()


if __name__ == '__main__':
    main()
