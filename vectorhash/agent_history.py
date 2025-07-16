import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import math
from fourier_scaffold import FourierScaffold
from graph_utils import (
    plot_probability_distribution_on_ax,
    plot_certainty_on_ax,
    error_test,
)
import torch


class VectorhashAgentHistory:
    def __init__(self):
        self._true_positions = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._certainty_odometry = []
        self._certainty_sensory = []

    def append(
        self,
        true_position,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
        certainty_odometry,
        certainty_sensory,
    ):
        self._true_positions.append(true_position.clone().cpu())
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
        theta_true_pos_artist = theta_dist_ax.plot(
            [self._true_positions[0][2]], [1.0], "ro"
        )
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
            theta_true_pos_artist[0].set_data([self._true_positions[frame][2]], [1.0])

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
            theta = self._true_positions[frame][2]
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
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._certainty_odometry = []
        self._certainty_sensory = []

    def calculate_errors(self):
        errors = torch.zeros(3, len(self._true_positions))  # (x/y/theta, N)

        for i in range(len(self._true_positions)):
            x_err = error_test(self._true_positions[i][0], self._x_distributions[i])
            y_err = error_test(self._true_positions[i][1], self._y_distributions[i])
            theta_err = error_test(
                self._true_positions[i][2], self._theta_distributions[i]
            )
            errors[0, i] = x_err
            errors[1, i] = y_err
            errors[2, i] = theta_err

        return errors


class VectorhashAgentKidnappedHistory:
    def __init__(self):
        self._true_positions = []
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
        theta_true_pos_artist = theta_dist_ax.plot(
            [self._true_positions[0][2]], [1.0], "ro"
        )
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
            theta_true_pos_artist[0].set_data([self._true_positions[frame][2]], [1.0])
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
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []
        self._seen = []
        self._certainty_odometry = []
        self._certainty_sensory = []

    def calculate_errors(self):
        errors = torch.zeros(3, len(self._true_positions))  # (x/y/theta, N)

        for i in range(len(self._true_positions)):
            x_err = error_test(self._true_positions[i][0], self._x_distributions[i])
            y_err = error_test(self._true_positions[i][1], self._y_distributions[i])
            theta_err = error_test(
                self._true_positions[i][2], self._theta_distributions[i]
            )
            errors[0, i] = x_err
            errors[1, i] = y_err
            errors[2, i] = theta_err

        return errors


class FourierVectorhashAgentHistory:
    def __init__(self) -> None:
        self._Ps = []
        self._true_images = []
        self._estimated_images = []
        self._Hs_odometry = []
        self._Hs_sensory = []
        self._true_positions = []

        self._xy_distributions = []
        self._th_distributions = []

        self.r_x = 5
        self.r_y = 5
        self.r_theta = 5
        self.ani = None

    def append(
        self,
        P: torch.Tensor | None,
        true_image: torch.Tensor,
        estimated_image: torch.Tensor | None,
        entropy_odometry: float | None,
        entropy_sensory: float | None,
        true_position: torch.Tensor,
        scaffold: FourierScaffold,
    ):
        self._true_images.append(true_image.clone().cpu())
        self._true_positions.append(true_position.clone().cpu())

        if (
            P != None
            and estimated_image != None
            and entropy_odometry
            and entropy_sensory
        ):
            self._Ps.append(P.clone().cpu())
            self._estimated_images.append(estimated_image.clone().cpu())
            self._Hs_odometry.append(entropy_odometry)
            self._Hs_sensory.append(entropy_sensory)

            x, y, theta = (
                torch.floor(true_position[0]),
                torch.floor(true_position[1]),
                torch.floor(true_position[2]),
            )
            xs = torch.arange(start=x - self.r_x, end=x + self.r_x + 1, device=P.device)  # type: ignore
            ys = torch.arange(start=y - self.r_y, end=y + self.r_y + 1, device=P.device)  # type: ignore
            thetas = torch.arange(start=theta - self.r_theta, end=theta + self.r_theta + 1, device=P.device)  # type: ignore

            # (N,d)
            omega = torch.cartesian_prod(xs, ys, thetas)

            # (D,M,d)**(d,N)->(D,M,N)->(D,N)->(D,D,N)
            encodings = scaffold.encode_batch(omega.T)

            # (D,D) x (D,D,N) -> (N)
            probabilities = torch.einsum("ij,ijb->b", P, encodings.conj()).abs()

            # (N) -> (N_x, N_y, N_theta)
            probabilities = probabilities.reshape(len(xs), len(ys), len(thetas))

            # (N_x, N_y, N_theta) -> (N_x, N_y)
            probabilities_xy = probabilities.sum(2)

            # (N_x, N_y, N_theta) -> (N_theta)
            probabilities_theta = probabilities.sum(0).sum(0)

            self._xy_distributions.append(probabilities_xy.cpu())
            self._th_distributions.append(probabilities_theta.cpu())
        else:
            self._Ps.append(None)
            self._estimated_images.append(None)
            self._Hs_odometry.append(None)
            self._Hs_sensory.append(None)
            self._xy_distributions.append(None)
            self._th_distributions.append(None)

    def reset(self):
        self.ani = None
        self._true_images = []
        self._estimated_images = []
        self._true_positions = []
        self._xy_distributions = []
        self._th_distributions = []
        self._Hs_odometry = []
        self._Hs_sensory = []

    def make_image_video(self):
        # 0         3 4       6
        #
        # +---------+---------+  0
        # |         |         |
        # |  true   |  pred   |
        # |   img   |   img   |
        # |         |         |
        # +---------+-+-------+  3
        # |  xy_dist  |th_dist|
        # |           |       |
        # |           |       |
        # |           |       |
        # |           |       |
        # +-----------+-------+  7
        # entropy_o:  1.22       8
        # entropy_s:  4.21
        #
        #
        fig = plt.figure(layout="constrained", figsize=(7, 7), dpi=100)
        gs = GridSpec(nrows=8, ncols=6, figure=fig)

        text_artist = fig.suptitle("t=0")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        xy_dist_ax = fig.add_subplot(gs[3:7, 0:4])
        th_dist_ax = fig.add_subplot(gs[3:7, 4:6])
        info_ax = fig.add_subplot(gs[7, 0:6])

        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        xy_dist_ax.set_title("xy dist around true pos")
        th_dist_ax.set_title("θ dist around true pos")
        info_ax.set_title("info")

        x, y, theta = (
            torch.floor(self._true_positions[0][0]).item(),
            torch.floor(self._true_positions[0][1]).item(),
            torch.floor(self._true_positions[0][2]).item(),
        )
        # xy_dist_ax.set_xlim(x - self.r_x, x + self.r_x)
        # xy_dist_ax.set_ylim(y - self.r_y, y + self.r_y)
        th_dist_ax.set_ylim(
            theta - self.r_theta,
            theta + self.r_theta,
        )
        info_ax.set_ylim(0, 1)

        im_true_artist = im_true_ax.imshow(self._true_images[0])
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0])

        extent = (
            x - self.r_x - 0.5,
            x + self.r_x + 1 - 0.5,
            y - self.r_y - 0.5,
            y + self.r_y + 1 - 0.5,
        )
        xy_dist_artist = xy_dist_ax.imshow(self._xy_distributions[0], extent=extent)
        th_dist_artist = plot_probability_distribution_on_ax(
            self._th_distributions[0],
            th_dist_ax,
            orientation="horizontal",
            start=theta - self.r_theta,
        )
        xy_true_pos_artist = xy_dist_ax.plot(
            [self._true_positions[0][0]], [self._true_positions[0][1]], "ro"
        )
        th_true_pos_artist = th_dist_ax.plot(
            [1.0], [self._true_positions[0][2] + 0.5], "ro"
        )
        entropy_artist = info_ax.text(
            0, 0, f"H_o: {self._Hs_odometry[0]:.3f}; H_s: {self._Hs_sensory[0]:.3f}"
        )

        def plot_func(frame):
            im_true_artist.set_data(self._true_images[frame])
            xy_true_pos_artist[0].set_data(
                [self._true_positions[frame][0]], [self._true_positions[frame][1]]
            )
            th_true_pos_artist[0].set_data(
                [1.0], [self._true_positions[frame][2] + 0.5]
            )
            text_artist.set_text(f"t={frame}")

            x, y, theta = (
                torch.floor(self._true_positions[frame][0]).item(),
                torch.floor(self._true_positions[frame][1]).item(),
                torch.floor(self._true_positions[frame][2]).item(),
            )
            # xy_dist_ax.set_xlim(x - self.r_x, x + self.r_x)
            # xy_dist_ax.set_ylim(y - self.r_y, y + self.r_y)
            th_dist_ax.set_ylim(
                theta - self.r_theta,
                theta + self.r_theta,
            )
            extent = (
                x - self.r_x - 0.5,
                x + self.r_x + 1 - 0.5,
                y - self.r_y - 0.5,
                y + self.r_y + 1 - 0.5,
            )
            xy_dist_artist.set_extent(extent)
            artists = [
                im_true_artist,
                xy_true_pos_artist,
                th_true_pos_artist,
                text_artist,
                xy_dist_ax,
                xy_dist_artist,
                th_dist_ax,
            ]

            if self._estimated_images[frame] != None:
                im_pred_artist.set_data(self._estimated_images[frame])
                xy_dist_artist.set_data(self._xy_distributions[frame])
                th_dist_artist.set_data(
                    values=self._th_distributions[frame],
                    edges=torch.arange(theta - self.r_theta, theta + self.r_theta + 2),
                )

                entropy_artist.set_text(
                    f"H_o: {self._Hs_odometry[frame]:.3f}; H_s: {self._Hs_sensory[frame]:.3f}"
                )

                artists.append(im_pred_artist)
                artists.append(th_dist_artist)
                artists.append(entropy_artist)

            return artists

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani
