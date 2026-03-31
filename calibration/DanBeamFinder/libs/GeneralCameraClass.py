import numpy as np
import matplotlib
# Use a non-Qt backend if your Linux machine is having Qt issues
# Comment this out if you are in a normal desktop session and want interactive windows
matplotlib.use("TkAgg")   # or use "Agg" if you only want to save figures

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
        self.shmFrame_beamFull = shm("/dev/shm/cred1.im.shm")

    def SharedMemorybeam1(self):
        self.shmFrame_beam1 = shm("/dev/shm/baldr1.im.shm")

    def SharedMemorybeam2(self):
        self.shmFrame_beam2 = shm("/dev/shm/baldr2.im.shm")

    def SharedMemorybeam3(self):
        self.shmFrame_beam3 = shm("/dev/shm/baldr3.im.shm")

    def SharedMemorybeam4(self):
        self.shmFrame_beam4 = shm("/dev/shm/baldr4.im.shm")
        
    def GetFrame(self, ibeam):
        if ibeam == 1:
            shmFrame = self.shmFrame_beam1
        elif ibeam == 2:
            shmFrame = self.shmFrame_beam2
        elif ibeam == 3:
            shmFrame = self.shmFrame_beam3
        elif ibeam == 4:
            shmFrame = self.shmFrame_beam4
        elif ibeam == 0:
            shmFrame = self.shmFrame_beamFull
        
        img_tmp = shmFrame.get_data()
        if len(img_tmp.shape) > 2:
            img_tmp = np.mean(img_tmp, axis=0)
        return img_tmp
    
    @staticmethod
    def apply_circular_aperture(array, center, radius, fill_value=0):
        rows, cols = array.shape
        y, x = np.ogrid[:rows, :cols]
        cy, cx = center

        mask = (x - cx)**2 + (y - cy)**2 >= radius**2

        masked_array = np.full_like(array, fill_value)
        masked_array[mask] = array[mask]
        return masked_array

    @staticmethod
    def center_of_mass(array):
        array = np.asarray(array, dtype=float)

        total = np.sum(array)
        if total == 0:
            raise ValueError("Array sum is zero; cannot compute centre of mass.")

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
        if frame is None:
            raise ValueError("Need to pass a frame")

        if avgCount > 1:
            framePwrTotal = 0
            for _ in range(avgCount):
                bg = np.nanmedian(frame)
                roi, _ = GeneralCameraObject.ApatrureFrame(
                    frame,
                    centre=centre,
                    x_half_width=x_half_width,
                    y_half_width=y_half_width,
                    show_plot=show_plot
                )
                framePwr = np.nansum(roi - bg)
                framePwrTotal += framePwr
            framePwr = framePwrTotal / avgCount
        else:
            bg = np.nanmedian(frame)
            roi, _ = GeneralCameraObject.ApatrureFrame(
                frame,
                centre=centre,
                x_half_width=x_half_width,
                y_half_width=y_half_width,
                show_plot=show_plot
            )
            framePwr = np.nansum(roi - bg)

        return framePwr

    @staticmethod
    def rescaleFrame_256(frame):
        lo = frame.min()
        hi = frame.max()
        if hi == lo:
            return np.zeros_like(frame, dtype=np.uint8)
        norm = (frame - lo) / (hi - lo)
        return (norm * 255).astype(np.uint8)

    @staticmethod
    def ApatrureFrame(frame, centre=None, x_half_width=None, y_half_width=None, show_plot=False):
        roi = frame

        if centre is not None and x_half_width is not None and y_half_width is not None:
            cy, cx = centre
            nrows, ncols = frame.shape

            y_start = max(int(cy - y_half_width), 0)
            y_end   = min(int(cy + y_half_width), nrows)
            x_start = max(int(cx - x_half_width), 0)
            x_end   = min(int(cx + x_half_width), ncols)

            roi = frame[y_start:y_end, x_start:x_end]

            if show_plot:
                GeneralCameraObject.show_frame_with_roi(
                    frame=frame,
                    centre=centre,
                    x_half_width=x_half_width,
                    y_half_width=y_half_width,
                    title="ROI"
                )

        return roi, None

    @staticmethod
    def show_frame_with_roi(
        frame,
        centre=None,
        x_half_width=None,
        y_half_width=None,
        title="Frame",
        cmap="gray",
        draw_cross=True
    ):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap=cmap, origin="upper")
        ax.set_title(title)
        ax.axis("off")

        if centre is not None and x_half_width is not None and y_half_width is not None:
            cy, cx = centre
            x1 = cx - x_half_width
            y1 = cy - y_half_width
            width = 2 * x_half_width
            height = 2 * y_half_width

            rect = Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)

            if draw_cross:
                ax.plot(cx, cy, marker="+", color="red", markersize=10, markeredgewidth=1.5)

        plt.show()
        # plt.close()
    @staticmethod
    def FindMaxValueOnFrame(frame):
        Maxidx= np.unravel_index(np.argmax(frame),frame.shape)
        return Maxidx
    @staticmethod
    def FindMinValueOnFrame(frame):
        Mindx= np.unravel_index(np.argmin(frame),frame.shape)
        return Mindx
    @staticmethod
    def PlotFrames(iframe, Framebuffer):
        fig, ax1 = plt.subplots()
        ax1.imshow(np.squeeze(Framebuffer[iframe, :, :]), cmap="gray", origin="upper")
        ax1.set_title("CameraFrames", fontsize=8)
        ax1.axis("off")
        plt.show()

    @staticmethod
    def DisplayWindow_GraphWithText(
        image: np.ndarray,
        text: str,
        cmap="gray",
        figsize=(8, 8)
    ):
        """
        Display an image with multi-line text underneath using matplotlib.
        """
        lines = text.split("\n")

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.imshow(image, cmap=cmap, origin="upper")
        ax_img.axis("off")

        ax_txt.axis("off")
        ax_txt.text(
            0.01, 0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace"
        )

        plt.show()
        # plt.show(block=False)

        block=False
        return fig

    @staticmethod
    def CameraDisplayWindow(
        Frame_int,
        FrameWidth,
        FrameHeight,
        ClipFrame,
        frameMinClip,
        frameMaxClip,
        DisplayPowerOverROI,
        ROI_visual_arr,
        scale,
        pixelFormat,
        opencvWindowName
    ):
        Frame_intFordisplay = GeneralCameraObject.rescaleFrame_256(Frame_int)

        if DisplayPowerOverROI:
            roi, _ = GeneralCameraObject.ApatrureFrame(
                Frame_int,
                centre=[ROI_visual_arr[0], ROI_visual_arr[1]],
                x_half_width=ROI_visual_arr[2],
                y_half_width=ROI_visual_arr[3],
                show_plot=False
            )
            framePwr = np.mean(roi)
            framePwr_std = np.std(roi)
        else:
            framePwr = 0
            framePwr_std = 0

        WindowSting = f"ROI Mean: {framePwr:.0f} Std: {framePwr_std:.0f}\n"

        if pixelFormat == "Mono8":
            WindowSting += f"Max Value on Frame: {100*np.max(Frame_int)/255:3.0f}%\n"
        elif pixelFormat == "Mono10":
            WindowSting += f"Max Value on Frame: {100*np.max(Frame_int)/1023:3.0f}%\n"
        elif pixelFormat == "Mono12":
            WindowSting += f"Max Value on Frame: {100*np.max(Frame_int)/4095:3.0f}%\n"
        elif pixelFormat == "Mono14":
            WindowSting += f"Max Value on Frame: {100*np.max(Frame_int)/16383:3.0f}%\n"
        elif pixelFormat == "Mono16":
            WindowSting += f"Max Value on Frame: {100*np.max(Frame_int)/65535:3.0f}%\n"
        else:
            WindowSting += f"Unsupported pixel format! Max Value on Frame: {np.max(Frame_int):.0f}\n"

        fig = plt.figure(figsize=(8 * scale, 8 * scale))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)

        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.imshow(Frame_intFordisplay, cmap="gray", origin="upper")
        ax_img.axis("off")
        ax_img.set_title(opencvWindowName)

        if DisplayPowerOverROI:
            cy, cx = ROI_visual_arr[0], ROI_visual_arr[1]
            x_half_width = ROI_visual_arr[2]
            y_half_width = ROI_visual_arr[3]

            rect = Rectangle(
                (cx - x_half_width, cy - y_half_width),
                2 * x_half_width,
                2 * y_half_width,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none"
            )
            ax_img.add_patch(rect)

        ax_txt.axis("off")
        ax_txt.text(
            0.01, 0.95,
            WindowSting,
            va="top",
            ha="left",
            fontsize=max(8, int(10 * scale)),
            family="monospace"
        )

        plt.show()
        # plt.show(block=False)

        return fig