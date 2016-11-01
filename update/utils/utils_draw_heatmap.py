def draw_heatmap(  heatmap):
    fig = plt.figure(1, figsize=(10,10))
    ax2 = fig.add_subplot(222)
    ax2.imshow(heatmap, origin='lower', aspect='auto')
    ax2.set_title("heatmap")
    plt.show()
