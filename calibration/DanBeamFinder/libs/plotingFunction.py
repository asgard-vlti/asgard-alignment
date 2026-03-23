import matplotlib.pyplot as plt
import numpy as np
def PlotResults(iscan,Ref_frame,AllFrames,gridpoints,MetricMatrix):
    
    
    CamImage=AllFrames[iscan,:,:];
    
    flat_points = gridpoints.reshape(-1, 2)
    x_plot = flat_points[:, 0]
    y_plot = flat_points[:, 1]
    colors = np.arange(len(flat_points))
    
    # Plot full path
    # Plot ONE highlighted point
    

    fig, ax1=plt.subplots(2,2);
    fig.subplots_adjust(wspace=0.1, hspace=0.1);

    ax1[0][0].plot(x_plot, y_plot, alpha=0.3)
    ax1[0][0].scatter(x_plot[iscan], y_plot[iscan], c=[colors[iscan]], cmap="viridis", s=100)
    ax1[0][0].set_title('Scan Profile',fontsize = 8);
    
    ax1[0][1].imshow(MetricMatrix);
    ax1[0][1].set_title('Metric Matrix',fontsize = 8);
    ax1[0][1].axis('off');
    
    
    ax1[1][0].imshow(Ref_frame);
    ax1[1][0].set_title('Ref Frame',fontsize = 8);
    ax1[1][0].axis('off');
    
    ax1[1][1].imshow(CamImage);
    ax1[1][1].set_title('Cam Image',fontsize = 8);
    ax1[1][1].axis('off')
    
    