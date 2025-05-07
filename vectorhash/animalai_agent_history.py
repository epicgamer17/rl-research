import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import math
from graph_utils import plot_probability_distribution_on_ax, plot_certainty_on_ax


class VectorhashAgentHistory:
    def __init__(self):
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._velocity_history = []

    def append(
        self,
        true_position,
        true_angle,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
    ):
        self._true_positions.append(true_position.clone().cpu())
        self._true_angles.append(true_angle)
        self._x_distributions.append(x_distribution.clone().cpu())
        self._y_distributions.append(y_distribution.clone().cpu())
        self._theta_distributions.append(theta_distribution.clone().cpu())
        self._true_images.append(true_image.clone().cpu())
        self._estimated_images.append(estimated_image.clone().cpu())

    def make_image_video(self):
        fig = plt.figure(layout="constrained", figsize=(6, 6), dpi=100)
        gs = GridSpec(6, 6, figure=fig)

        text_artist = fig.suptitle("t=0")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        x_dist_ax = fig.add_subplot(gs[3, 0:4])
        y_dist_ax = fig.add_subplot(gs[4, 0:4])
        theta_dist_ax = fig.add_subplot(gs[5, 0:4])
        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        x_dist_ax.set_title("x dist")
        y_dist_ax.set_title("y dist")
        theta_dist_ax.set_title("θ dist")

        x_dist_ax.set_xlim(0, len(self._x_distributions[0]))
        y_dist_ax.set_xlim(0, len(self._y_distributions[0]))
        theta_dist_ax.set_xlim(0, len(self._theta_distributions[0]))

        im_true_artist = im_true_ax.imshow(self._true_images[0], vmin=0, vmax=1)
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0], vmin=0, vmax=1)
        x_dist_artist = plot_probability_distribution_on_ax(
            self._x_distributions[0], x_dist_ax
        )
        y_dist_artist = plot_probability_distribution_on_ax(
            self._y_distributions[0], y_dist_ax
        )
        theta_dist_artist = plot_probability_distribution_on_ax(
            self._theta_distributions[0], theta_dist_ax
        )

        x_true_pos_artist = x_dist_ax.plot([self._true_positions[0][0]], [1.0], "ro")
        y_true_pos_artist = y_dist_ax.plot([self._true_positions[0][1]], [1.0], "ro")
        theta_true_pos_artist = theta_dist_ax.plot([self._true_angles[0]], [1.0], "ro")

        def plot_func(frame):
            im_true_artist.set_data(self._true_images[frame])
            im_pred_artist.set_data(self._estimated_images[frame])

            x_dist_artist.set_data(values=self._x_distributions[frame], edges=None)
            y_dist_artist.set_data(values=self._y_distributions[frame], edges=None)
            theta_dist_artist.set_data(
                values=self._theta_distributions[frame], edges=None
            )

            x_true_pos_artist[0].set_data([self._true_positions[frame][0]], [1.0])
            y_true_pos_artist[0].set_data([self._true_positions[frame][1]], [1.0])
            theta_true_pos_artist[0].set_data([self._true_angles[frame]], [1.0])

            text_artist.set_text(f"t={frame}")
            return (
                im_true_artist,
                im_pred_artist,
                x_dist_artist,
                y_dist_artist,
                theta_dist_artist,
                text_artist,
                x_true_pos_artist,
                y_true_pos_artist,
                theta_true_pos_artist,
            )

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def plot_vector_position(self):
        fig = plt.figure(layout="constrained", figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Vector Position")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal")

        def plot_func(frame):
            x = self._true_positions[frame][0]
            y = self._true_positions[frame][1]
            theta = self._true_angles[frame]
            dx = math.cos(theta) * 2
            dy = math.sin(theta) * 2
            ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1, color="r")
            ax.plot(x, y, "ro")

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def reset(self):
        self.ani = None
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []


class VectorhashAgentKidnappedHistory:
    def __init__(self):
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._seen = []

    def append(
        self,
        true_position,
        true_angle,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
        seen,
    ):
        self._true_positions.append(true_position.clone().cpu())
        self._true_angles.append(true_angle)
        self._true_images.append(true_image.clone().cpu())
        self._seen.append(seen)
        self._estimated_images.append(
            estimated_image.clone().cpu() if estimated_image is not None else None
        )
        self._x_distributions.append(
            x_distribution.clone().cpu() if x_distribution is not None else None
        )
        self._y_distributions.append(
            y_distribution.clone().cpu() if y_distribution is not None else None
        )
        self._theta_distributions.append(
            theta_distribution.clone().cpu() if theta_distribution is not None else None
        )

    def make_image_video(self):
        fig = plt.figure(layout="constrained", figsize=(6, 6), dpi=100)
        gs = GridSpec(6, 6, figure=fig)

        text_artist = fig.suptitle("t=0, seen=True")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        x_dist_ax = fig.add_subplot(gs[3, 0:4])
        y_dist_ax = fig.add_subplot(gs[4, 0:4])
        theta_dist_ax = fig.add_subplot(gs[5, 0:4])
        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        x_dist_ax.set_title("x dist")
        y_dist_ax.set_title("y dist")
        theta_dist_ax.set_title("θ dist")

        x_dist_ax.set_xlim(0, len(self._x_distributions[0]))
        y_dist_ax.set_xlim(0, len(self._y_distributions[0]))
        theta_dist_ax.set_xlim(0, len(self._theta_distributions[0]))

        im_true_artist = im_true_ax.imshow(self._true_images[0], vmin=0, vmax=1)
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0], vmin=0, vmax=1)
        x_dist_artist = plot_probability_distribution_on_ax(
            self._x_distributions[0], x_dist_ax
        )
        y_dist_artist = plot_probability_distribution_on_ax(
            self._y_distributions[0], y_dist_ax
        )
        theta_dist_artist = plot_probability_distribution_on_ax(
            self._theta_distributions[0], theta_dist_ax
        )

        x_true_pos_artist = x_dist_ax.plot([self._true_positions[0][0]], [1.0], "ro")
        y_true_pos_artist = y_dist_ax.plot([self._true_positions[0][1]], [1.0], "ro")
        theta_true_pos_artist = theta_dist_ax.plot([self._true_angles[0]], [1.0], "ro")

        def plot_func(frame):
            artists = []
            im_true_artist.set_data(self._true_images[frame])
            artists.append(im_true_artist)

            if self._estimated_images[frame] is not None:
                im_pred_artist.set_data(self._estimated_images[frame])
                artists.append(im_pred_artist)

            if self._x_distributions[frame] is not None:
                x_dist_artist.set_data(values=self._x_distributions[frame], edges=None)
                artists.append(x_dist_artist)

            if self._y_distributions[frame] is not None:
                y_dist_artist.set_data(values=self._y_distributions[frame], edges=None)
                artists.append(y_dist_artist)

            if self._theta_distributions[frame] is not None:
                theta_dist_artist.set_data(
                    values=self._theta_distributions[frame], edges=None
                )
                artists.append(theta_dist_artist)

            x_true_pos_artist[0].set_data([self._true_positions[frame][0]], [1.0])
            y_true_pos_artist[0].set_data([self._true_positions[frame][1]], [1.0])
            theta_true_pos_artist[0].set_data([self._true_angles[frame]], [1.0])

            artists.append(x_true_pos_artist)
            artists.append(y_true_pos_artist)
            artists.append(theta_true_pos_artist)

            text_artist.set_text(f"t={frame}, seen={self._seen[frame]}")
            return artists

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def reset(self):
        self.ani = None
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._seen = []


class VectorhashAgentHistoryWithCertainty:
    def __init__(self):
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._velocity_history = []

        self._certainty_odometry = []
        self._certainty_sensory = []

    def append(
        self,
        true_position,
        true_angle,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
        certainty_odometry,
        certainty_sensory,
    ):
        self._true_positions.append(true_position.clone().cpu())
        self._true_angles.append(true_angle)
        self._x_distributions.append(x_distribution.clone().cpu())
        self._y_distributions.append(y_distribution.clone().cpu())
        self._theta_distributions.append(theta_distribution.clone().cpu())
        self._true_images.append(true_image.clone().cpu())
        self._estimated_images.append(estimated_image.clone().cpu())
        self._certainty_odometry.append(certainty_odometry.clone().cpu())
        self._certainty_sensory.append(certainty_sensory.clone().cpu())

    def make_image_video(self):
        fig = plt.figure(layout="constrained", figsize=(7, 7), dpi=100)
        gs = GridSpec(6, 6, figure=fig)

        text_artist = fig.suptitle("t=0")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        x_dist_ax = fig.add_subplot(gs[3, 0:4])
        y_dist_ax = fig.add_subplot(gs[4, 0:4])
        theta_dist_ax = fig.add_subplot(gs[5, 0:4])
        certainty_ax = fig.add_subplot(gs[3:6, 4:6])

        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        x_dist_ax.set_title("x dist")
        y_dist_ax.set_title("y dist")
        theta_dist_ax.set_title("θ dist")
        certainty_ax.set_title("odometric and sensory certainties")

        x_dist_ax.set_xlim(0, len(self._x_distributions[0]))
        y_dist_ax.set_xlim(0, len(self._y_distributions[0]))
        theta_dist_ax.set_xlim(0, len(self._theta_distributions[0]))
        certainty_ax.set_ylim(0, 1)

        im_true_artist = im_true_ax.imshow(self._true_images[0], vmin=0, vmax=1)
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0], vmin=0, vmax=1)
        x_dist_artist = plot_probability_distribution_on_ax(
            self._x_distributions[0], x_dist_ax
        )
        y_dist_artist = plot_probability_distribution_on_ax(
            self._y_distributions[0], y_dist_ax
        )
        theta_dist_artist = plot_probability_distribution_on_ax(
            self._theta_distributions[0], theta_dist_ax
        )
        x_true_pos_artist = x_dist_ax.plot([self._true_positions[0][0]], [1.0], "ro")
        y_true_pos_artist = y_dist_ax.plot([self._true_positions[0][1]], [1.0], "ro")
        theta_true_pos_artist = theta_dist_ax.plot([self._true_angles[0]], [1.0], "ro")
        certainty_artists = plot_certainty_on_ax(
            certainty_odometry=self._certainty_odometry[0],
            certainty_sensory=self._certainty_sensory[0],
            ax=certainty_ax,
        )

        def plot_func(frame):
            im_true_artist.set_data(self._true_images[frame])
            im_pred_artist.set_data(self._estimated_images[frame])

            x_dist_artist.set_data(values=self._x_distributions[frame], edges=None)
            y_dist_artist.set_data(values=self._y_distributions[frame], edges=None)
            theta_dist_artist.set_data(
                values=self._theta_distributions[frame], edges=None
            )

            x_true_pos_artist[0].set_data([self._true_positions[frame][0]], [1.0])
            y_true_pos_artist[0].set_data([self._true_positions[frame][1]], [1.0])
            theta_true_pos_artist[0].set_data([self._true_angles[frame]], [1.0])

            certainty_artists[0].set_height(self._certainty_odometry[frame][0])
            certainty_artists[1].set_height(self._certainty_odometry[frame][1])
            certainty_artists[2].set_height(self._certainty_odometry[frame][2])
            certainty_artists[3].set_height(self._certainty_sensory[frame][0])
            certainty_artists[4].set_height(self._certainty_sensory[frame][1])
            certainty_artists[5].set_height(self._certainty_sensory[frame][2])

            text_artist.set_text(f"t={frame}")
            return (
                im_true_artist,
                im_pred_artist,
                x_dist_artist,
                y_dist_artist,
                theta_dist_artist,
                text_artist,
                x_true_pos_artist,
                y_true_pos_artist,
                theta_true_pos_artist,
            ) + tuple(certainty_artists)

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def plot_vector_position(self):
        fig = plt.figure(layout="constrained", figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Vector Position")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal")

        def plot_func(frame):
            x = self._true_positions[frame][0]
            y = self._true_positions[frame][1]
            theta = self._true_angles[frame]
            dx = math.cos(theta) * 2
            dy = math.sin(theta) * 2
            ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1, color="r")
            ax.plot(x, y, "ro")

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def reset(self):
        self.ani = None
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._certainty_odometry = []
        self._certainty_sensory = []


class VectorhashAgentKidnappedHistoryWithCertainty:
    def __init__(self):
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._seen = []
        self._certainty_odometry = []
        self._certainty_sensory = []

    def append(
        self,
        true_position,
        true_angle,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
        seen,
        certainty_odometry,
        certainty_sensory,
    ):
        self._true_positions.append(true_position.clone().cpu())
        self._true_angles.append(true_angle)
        self._true_images.append(true_image.clone().cpu())
        self._seen.append(seen)
        self._estimated_images.append(
            estimated_image.clone().cpu() if estimated_image is not None else None
        )
        self._x_distributions.append(
            x_distribution.clone().cpu() if x_distribution is not None else None
        )
        self._y_distributions.append(
            y_distribution.clone().cpu() if y_distribution is not None else None
        )
        self._theta_distributions.append(
            theta_distribution.clone().cpu() if theta_distribution is not None else None
        )
        self._certainty_odometry.append(
            certainty_odometry.clone().cpu() if certainty_odometry is not None else None
        )
        self._certainty_sensory.append(
            certainty_sensory.clone().cpu() if certainty_sensory is not None else None
        )

    def make_image_video(self):
        fig = plt.figure(layout="constrained", figsize=(7, 7), dpi=100)
        gs = GridSpec(6, 6, figure=fig)

        text_artist = fig.suptitle("t=0, seen=True")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        x_dist_ax = fig.add_subplot(gs[3, 0:4])
        y_dist_ax = fig.add_subplot(gs[4, 0:4])
        theta_dist_ax = fig.add_subplot(gs[5, 0:4])
        certainty_ax = fig.add_subplot(gs[3:6, 4:6])

        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        x_dist_ax.set_title("x dist")
        y_dist_ax.set_title("y dist")
        theta_dist_ax.set_title("θ dist")

        x_dist_ax.set_xlim(0, len(self._x_distributions[0]))
        y_dist_ax.set_xlim(0, len(self._y_distributions[0]))
        theta_dist_ax.set_xlim(0, len(self._theta_distributions[0]))

        im_true_artist = im_true_ax.imshow(self._true_images[0], vmin=0, vmax=1)
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0], vmin=0, vmax=1)
        x_dist_artist = plot_probability_distribution_on_ax(
            self._x_distributions[0], x_dist_ax
        )
        y_dist_artist = plot_probability_distribution_on_ax(
            self._y_distributions[0], y_dist_ax
        )
        theta_dist_artist = plot_probability_distribution_on_ax(
            self._theta_distributions[0], theta_dist_ax
        )

        x_true_pos_artist = x_dist_ax.plot([self._true_positions[0][0]], [1.0], "ro")
        y_true_pos_artist = y_dist_ax.plot([self._true_positions[0][1]], [1.0], "ro")
        theta_true_pos_artist = theta_dist_ax.plot([self._true_angles[0]], [1.0], "ro")
        certainty_artists = plot_certainty_on_ax(
            certainty_odometry=self._certainty_odometry[0],
            certainty_sensory=self._certainty_sensory[0],
            ax=certainty_ax,
        )

        def plot_func(frame):
            artists = []
            im_true_artist.set_data(self._true_images[frame])
            artists.append(im_true_artist)

            if self._estimated_images[frame] is not None:
                im_pred_artist.set_data(self._estimated_images[frame])
                artists.append(im_pred_artist)

            if self._x_distributions[frame] is not None:
                x_dist_artist.set_data(values=self._x_distributions[frame], edges=None)
                artists.append(x_dist_artist)

            if self._y_distributions[frame] is not None:
                y_dist_artist.set_data(values=self._y_distributions[frame], edges=None)
                artists.append(y_dist_artist)

            if self._theta_distributions[frame] is not None:
                theta_dist_artist.set_data(
                    values=self._theta_distributions[frame], edges=None
                )
                artists.append(theta_dist_artist)

            if self._certainty_odometry[frame] is not None:
                certainty_artists[0].set_height(self._certainty_odometry[frame][0])
                certainty_artists[1].set_height(self._certainty_odometry[frame][1])
                certainty_artists[2].set_height(self._certainty_odometry[frame][2])
                certainty_artists[3].set_height(self._certainty_sensory[frame][0])
                certainty_artists[4].set_height(self._certainty_sensory[frame][1])
                certainty_artists[5].set_height(self._certainty_sensory[frame][2])

                artists += certainty_artists

            x_true_pos_artist[0].set_data([self._true_positions[frame][0]], [1.0])
            y_true_pos_artist[0].set_data([self._true_positions[frame][1]], [1.0])
            theta_true_pos_artist[0].set_data([self._true_angles[frame]], [1.0])
            artists += [x_true_pos_artist, y_true_pos_artist, theta_true_pos_artist]

            text_artist.set_text(f"t={frame}, seen={self._seen[frame]}")
            artists.append(text_artist)
            return artists

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def reset(self):
        self.ani = None
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._seen = []
        self._certainty_odometry = []
        self._certainty_sensory = []
