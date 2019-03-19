import numpy as np


class Tracker():

    def __init__(self, win_width, win_height, margin, xm=1, ym=1, smooth_factor=15):
        self.recent_centers = []
        self.win_width = win_width
        self.win_height = win_height
        self.margin = margin
        self.xm_per_pix = xm    # Meters per pixel along horizontal axis
        self.ym_per_pix = ym    # Meters per pixel along vertical axis
        self.smooth_factor = smooth_factor

    def find_window_centroids(self, image):
        win_width = self.win_width
        win_height = self.win_height
        margin = self.margin

        # Store the (left, right) window centroid positions per level
        win_centroids = []

        # Create our win template that we will use for convolutions
        win = np.ones(win_width)

        # First find the two starting positions for the left and right lane by
        # using np.sum to get the vertical image slice and then np.convolve the
        # vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(win, l_sum)) - win_width / 2
        r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(win, r_sum)) - win_width / 2 + int(image.shape[1] / 2)

        # Add what we found for the first layer
        win_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(image.shape[0] / win_height)):
            # Convolve the window into the vertical slice of the image
            image_layer = np.sum(
                image[int(image.shape[0] - (level + 1) * win_height):int(image.shape[0] - level * win_height), :],
                axis=0)
            conv_signal = np.convolve(win, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use win_width/2 as offset because convolution signal reference is
            # at right side of win, not center of win
            offset = win_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            win_centroids.append((l_center, r_center))

        self.recent_centers.append(win_centroids)

        # Return average values of the line centers, avoid markers from jumping around too much
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
