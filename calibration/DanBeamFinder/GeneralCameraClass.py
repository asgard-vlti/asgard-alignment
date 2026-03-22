import numpy as np
# import math as cv2
import cv2
import matplotlib.pyplot as plt
from xaosim.shmlib import shm 

class GeneralCameraObject():
    def __init__(self):
        pass
        self.SharedMemoryFullFrame()
        self.SharedMemorybeam1()
        self.SharedMemorybeam2()
        self.SharedMemorybeam3()
        self.SharedMemorybeam4()


    
    def SharedMemoryFullFrame(self):
        self.shmFrame_beamFull=shm( f"/dev/shm/cred1.im.shm")
    def SharedMemorybeam1(self):
        self.shmFrame_beam1= shm( f"/dev/shm/baldr1.im.shm")
    def SharedMemorybeam2(self):
        self.shmFrame_beam2= shm( f"/dev/shm/baldr2.im.shm")
    def SharedMemorybeam3(self):
        self.shmFrame_beam3= shm( f"/dev/shm/baldr3.im.shm")
    def SharedMemorybeam4(self):
        self.shmFrame_beam4= shm( f"/dev/shm/baldr4.im.shm")
        
    def GetFrame(self, ibeam):
        if ibeam==1:
            shmFrame=self.shmFrame_beam1
        elif ibeam==2:
            shmFrame=self.shmFrame_beam2
        elif ibeam==3:
            shmFrame=self.shmFrame_beam3
        elif ibeam==4:
            shmFrame=self.shmFrame_beam4    
        if ibeam==0:
                shmFrame=self.shmFrame_beamFull
        
        img_tmp =  shmFrame.get_data() 
        if len( img_tmp.shape )>2:
            img_tmp = np.mean( shmFrame.get_data() ,axis=0)
        return img_tmp
    
    @staticmethod
    def apply_circular_aperture(array, center, radius, fill_value=0):
        """
        Apply a circular aperture to a 2D numpy array.
        
        Parameters:
        -----------
        array : np.ndarray
            Input 2D array (e.g., image or data field).
        center : tuple of (float, float)
            (row, col) coordinates of the circle centre.
        radius : float
            Radius of the circular aperture (in pixels).
        fill_value : number, optional
            Value to assign outside the aperture (default = 0).
        
        Returns:
        --------
        masked_array : np.ndarray
            Array with circular aperture applied.
        """
        rows, cols = array.shape
        y, x = np.ogrid[:rows, :cols]
        
        cy, cx = center
        # mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        mask = (x - cx)**2 + (y - cy)**2 >= radius**2

        
        masked_array = np.full_like(array, fill_value)
        masked_array[mask] = array[mask]
        return masked_array
    @staticmethod
    def center_of_mass(array):
        """
        Compute the center of mass of a 2D numpy array.
        
        Parameters
        ----------
        array : np.ndarray
            2D array of values (weights). Must be non-negative if you want
            a meaningful "centre of mass".
        
        Returns
        -------
        (cy, cx) : tuple of floats
            The (row, col) coordinates of the centre of mass.
        """
        array = np.asarray(array, dtype=float)

        total = np.sum(array)
        if total == 0:
            raise ValueError("Array sum is zero; cannot compute centre of mass.")

        # coordinate grids
        rows, cols = array.shape
        y, x = np.arange(rows), np.arange(cols)
        X, Y = np.meshgrid(x, y)

        cx = np.sum(X * array) / total
        cy = np.sum(Y * array) / total

        return int(cy), int(cx)
    
    @staticmethod
    def GetRelativePower(
            frame=None,
            centre=None,
            x_half_width=None,
            y_half_width=None,
            show_plot=False,
            avgCount=1
        ):
            """
            Compute the relative power of a camera frame, optionally limited to a
            rectangular region of interest (ROI) and optionally display that region.

            Parameters
            ----------
            frame : 2D np.ndarray or None
                If None, grabs a frame with self.GetFrame().
            centre : tuple or None
                (row, col) = (y, x) index of the ROI centre.
                If None, uses the full frame.
            x_half_width : int or None
                Half-width of ROI in x (columns).
            y_half_width : int or None
                Half-width of ROI in y (rows).
            show_plot : bool, default False
                If True, plots the frame with the ROI rectangle drawn.

            Returns
            -------
            float
                Sum over (ROI - median_of_frame), ignoring NaNs.
            """
            if avgCount>1:
                framePwrTotal=0
                for iavg in range(avgCount):
                    if frame is None:
                        print("need to pass a frame")

                    # Remove background using median of full frame
                    bg = np.nanmedian(frame)

                    # Default: full-frame power
                    # roi = frame
                    # framePwr = np.nansum(roi - bg)
                    roi,_ = GeneralCameraObject.ApatrureFrame(frame,centre=centre,x_half_width=x_half_width,y_half_width=y_half_width,show_plot=show_plot)
                    framePwr = np.nansum(roi - bg)
                    framePwrTotal=framePwrTotal+framePwr
                framePwr=framePwrTotal/avgCount
            else:
                if frame is None:
                    print("need to pass a frame")
                    # frame = self.GetFrame()

                # Remove background using median of full frame
                bg = np.nanmedian(frame)

                # Default: full-frame power
                # roi = frame
                # framePwr = np.nansum(roi - bg)
                roi,_ = GeneralCameraObject.ApatrureFrame(frame,centre=centre,x_half_width=x_half_width,y_half_width=y_half_width,show_plot=show_plot)
                framePwr = np.nansum(roi - bg)
            
       

            return framePwr
    @staticmethod
    def draw_roi_box(
            image: np.ndarray,
            centre=None,
            x_half_width=None,
            y_half_width=None,
            colour=(255, 0, 0),   # red in BGR
        thickness=4,
        draw_cross=True,
    ):
        """
        Return a copy of `image` with an ROI rectangle (and optional centre cross) drawn.
        Compatible with DisplayWindow_GraphWithText because it returns a normal image array.

        Parameters
        ----------
        image : np.ndarray
            (Ny, Nx) uint8/uint16 or (Ny, Nx, 3) BGR.
        centre : (cy, cx) or None
        x_half_width, y_half_width : int/float or None
        colour : (B, G, R)
        thickness : int
        draw_cross : bool
        """
        if image.ndim == 2:
            # convert to BGR for coloured drawing
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            vis = image.copy()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # If ROI not specified, return image as-is (copy/converted)
        if centre is None or x_half_width is None or y_half_width is None:
            return vis

        cy, cx = centre
        h, w = vis.shape[:2]

        # bounds (clip to image)
        x1 = max(int(round(cx - x_half_width)), 0)
        x2 = min(int(round(cx + x_half_width)), w - 1)
        y1 = max(int(round(cy - y_half_width)), 0)
        y2 = min(int(round(cy + y_half_width)), h - 1)

        # draw rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thickness)

        # draw centre cross
        if draw_cross:
            cx_i = int(round(cx))
            cy_i = int(round(cy))
            if 0 <= cx_i < w and 0 <= cy_i < h:
                cv2.drawMarker(vis, (cx_i, cy_i), colour,
                            markerType=cv2.MARKER_CROSS,
                            markerSize=max(10, 2*thickness + 8),
                            thickness=thickness)

        return vis
    @staticmethod
    def rescaleFrame_256(frame):
        lo=frame.min()
        hi=frame.max()
        
        norm = (frame - lo) / (hi - lo)
        return (norm * 255).astype(np.uint8)
    @staticmethod
    def ApatrureFrame(frame,
                centre=None,
                x_half_width=None,
                y_half_width=None,show_plot=False):
                
                
                # Default: full-frame power
                roi = frame
                ROIPlot=0
                # If ROI specified, restrict to it
                if centre is not None and x_half_width is not None and y_half_width is not None:
                    cy, cx = centre
                    nrows, ncols = frame.shape

                    # Compute ROI bounds and clip to frame
                    y_start = max(int(cy - y_half_width), 0)
                    y_end   = min(int(cy + y_half_width), nrows)
                    x_start = max(int(cx - x_half_width), 0)
                    x_end   = min(int(cx + x_half_width), ncols)

                    roi = frame[y_start:y_end, x_start:x_end]
                    if show_plot:
                        imageForPlot=GeneralCameraObject.rescaleFrame_256(frame)
                        ROIPlot=GeneralCameraObject.draw_roi_box(
                        image=imageForPlot,
                        centre=centre,
                        x_half_width=x_half_width,
                        y_half_width=y_half_width,
                        colour=(255,0,0),   # red in BGR
                        thickness=1,
                        draw_cross=True)
                        
                        plt.imshow(ROIPlot)
                        plt.show()
                            
                return roi,ROIPlot
                
    @staticmethod             
    def PlotFrames(iframe,Framebuffer):
            fig, ax1=plt.subplots();
            # fig.subplots_adjust(wspace=0.1, hspace=-0.6);
            # ax1.cmplxplt.complexColormap(frame);
            ax1.imshow(np.squeeze(Framebuffer[iframe,:,:]),cmap='gray');
            ax1.set_title('CameraFrames',fontsize = 8);
            ax1.axis('off'); 
    @staticmethod      
    def DisplayWindow_GraphWithText( image: np.ndarray,
                                    text: str,
                                    font=cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale=1,
                                    thickness=2,
                                    text_color=(255, 255, 255),
                                    bg_color=(0, 0, 0),
                                    margin=10,
                                    line_spacing=5):
            """
            Displays an image at the top with multi-line text at the bottom.
            
            Parameters:
            image (np.ndarray): Image in BGR format.
            text (str): Multi-line text with lines separated by '\n'.
            font: OpenCV font.
            font_scale (float): Scale factor for the text.
            thickness (int): Text thickness.
            text_color (tuple): Text color in BGR.
            bg_color (tuple): Background color for the text area.
            margin (int): Margin around the text.
            line_spacing (int): Extra spacing between lines.
            """

            # Split the text into lines
            # Split the text into lines
            lines = text.split('\n')

            # Maximum allowed width for text
            max_allowed_width = image.shape[1] - 2 * margin

            # Measure widths at font_scale = 1
            ref_scale = 1.0
            ref_sizes = [cv2.getTextSize(line, font, ref_scale, thickness)[0][0] for line in lines]
            max_ref_width = max(ref_sizes) if lines else 1

            # Compute scale factor so text fits width
            font_scale = min(font_scale, max_allowed_width / max_ref_width)


            # Get text sizes for each line
            line_info = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
            line_sizes = [info[0] for info in line_info]  # (width, height) for each line
            line_heights = [size[1] for size in line_sizes]  # Heights of each text line

            # Calculate text area size
            max_text_width = max(size[0] for size in line_sizes) if lines else 0
            total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing + 2 * margin

            # Determine canvas size (image on top, text at the bottom)
            canvas_width = max(image.shape[1], max_text_width + 2 * margin)
            canvas_height = image.shape[0] + total_text_height

            # Ensure the image is in BGR (if grayscale, convert to 3-channel)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Create a blank canvas
            canvas = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)

            # Place the image at the top
            image_x = (canvas_width - image.shape[1]) // 2  # Center horizontally
            canvas[0:image.shape[0], image_x:image_x+image.shape[1]] = image

            # Draw text at the bottom
            text_y_start = image.shape[0] + margin  # Start text after the image
            for i, line in enumerate(lines):
                text_y = text_y_start + sum(line_heights[:i]) + i * line_spacing + line_heights[i]
                cv2.putText(canvas, line, (margin, text_y), font, font_scale, text_color, thickness)

            return canvas
        
    @staticmethod
    def CameraDisplayWindow(Frame_int,FrameWidth,FrameHeight,
                            ClipFrame,frameMinClip,frameMaxClip,
                            DisplayPowerOverROI,ROI_visual_arr,
                            scale,pixelFormat,opencvWindowName):
        
        Frame_intFordisplay = GeneralCameraObject.rescaleFrame_256(Frame_int)
                    
        # if ClipFrame:
        #     Frame_intFordisplay =clip_frame(Frame_intFordisplay,
        #                                 vmin_percent=frameMinClip,
        #                                 vmax_percent=frameMaxClip)
            # Frame_intFordisplay = CamForm.rescaleFrame_256(Frame_int)
        # else:
            # Frame_intFordisplay=np.copy(Frame_int)
        

        if (DisplayPowerOverROI):
            # bg = np.nanmedian(Frame_int)
            roi,_ = GeneralCameraObject,GeneralCameraObject.ApatrureFrame(Frame_int,centre=[ROI_visual_arr[0],ROI_visual_arr[1]]
                                            ,x_half_width=ROI_visual_arr[2],
                                            y_half_width=ROI_visual_arr[3],show_plot=False)
            # framePwr = np.nansum(roi - bg)
            # framePwr = np.nansum(roi)/(ROI_visual_arr[2]*ROI_visual_arr[3])
            framePwr = np.mean(roi)#np.nansum(roi)/(np.prod(roi.shape))
            framePwr_std = np.std(roi)


            Frame_intFordisplay=GeneralCameraObject.draw_roi_box(Frame_intFordisplay,centre=[ROI_visual_arr[0],ROI_visual_arr[1]]
                                            ,x_half_width=ROI_visual_arr[2],
                                            y_half_width=ROI_visual_arr[3],
                                            thickness=int(1),
                                            colour=(0,0,255),draw_cross=False)
        else:
            framePwr=0
            framePwr_std=0
        if scale!=1:
            Frame_intFordisplay = cv2.resize(Frame_intFordisplay, 
                                            (int(FrameWidth*scale), int(FrameHeight*scale)), interpolation=cv2.INTER_LINEAR)
        # WindowSting=f"ROI Sum: {np.sum(frame)/np.prod(frame.shape):.0f}\n"
        WindowSting=f"ROI Mean: {framePwr:.0f} Std: {framePwr_std:.0f}\n"
        if pixelFormat=="Mono8":
            WindowSting=WindowSting+ f"Max Value on Frame: {100*np.max(Frame_int)/255:3.0f}%\n"
        elif pixelFormat=="Mono10":
            WindowSting=WindowSting+ f"Max Value on Frame: {100*np.max(Frame_int)/1023:3.0f}%\n"
        elif pixelFormat=="Mono12":
            WindowSting=WindowSting+ f"Max Value on Frame: {100*np.max(Frame_int)/4095:3.0f}%\n"
        elif pixelFormat=="Mono14":
            WindowSting=WindowSting+ f"Max Value on Frame: {100*(np.max(Frame_int)/16383):3.0f}%\n"
        elif pixelFormat=="Mono16":
            WindowSting=WindowSting+ f"Max Value on Frame: {100*np.max(Frame_int)/65535:3.0f}%\n"
        else:
            WindowSting=WindowSting+ f"Unsuporrted pixel format! Max Value on Frame: {np.max(Frame_int):.0f}%\n"
        canvasToDispla_viewPort=GeneralCameraObject.DisplayWindow_GraphWithText(Frame_intFordisplay,WindowSting,thickness=int(1),font_scale=5*scale)

        # cv2.imshow(opencvWindowName, canvasToDispla_viewPort)
        plt.imshow(canvasToDispla_viewPort)
        