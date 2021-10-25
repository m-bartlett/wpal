from image import *
from kmeans import kmeans


def generate_ANSI_palette_from_pixels( *,
                                       rgb_pixels,
                                       kmeans_initial_colors,
                                       kmeans_iterations = 3,
                                       minimum_contrast = 0,
                                       light_palette = False,
                                       value_scalars = [0.9, 0.95, 1.25, 1.5],
                                       saturation_scalars = [1.5, 1.25, 1.0, 0.95],
                                       verbose = False ):

	rgb_pixels = filter_colors_in_ellipsoid_volume(
	  rgb_pixels,
	  ellipsoids = [
	      {   # Filter brown
	          "radii":  np.array([100,100,150]),
	          "offset": np.array([128, 128, 0])
	      },
	  ]
	)

	kmeans_cluster_centers, indices_sorted_by_neighbor_quantity = (
		kmeans(kmeans_initial_colors, rgb_pixels, kmeans_iterations)
	)

	initial_palette = (
	  constrain_background_colors_to_minimum_distance_from_target(
	    kmeans_cluster_centers,
	    constraints = [
	      {"color":0, "max_distance":10000},
	      {"color":7, "max_distance":5000},
	    ]
	  )
	)

	if light_palette:
	  initial_palette[[0,7]] = initial_palette[[7,0]] # Swap white and black, i.e. fg with bg


	if minimum_contrast:
	  foreground_colors = initial_palette[1:-1]
	  background_color  = initial_palette[0]
	  higher_contrast_foreground_colors = (
	    constrain_contrast_between_foreground_and_background_colors(
	      foreground_colors = foreground_colors,
	      background_color = background_color,
	      minimum_contrast = minimum_contrast,
	      verbose = verbose
	    )
	  )
	  initial_palette[1:-1] = higher_contrast_foreground_colors


	hsv_palette = rgb_palette_to_hsv_palette(initial_palette)

	highlight_index = get_most_saturated_color_index(hsv_palette)

	# value_scalars = [0.9, 0.95, 1.25, 1.5]
	# saturation_scalars = (
	#   [1.5, 1.25, 1.0, 0.95] if light_palette
	#   else [1, 1, 1, 1]
	# )

	palettes = (
	  create_gradated_palettes(
	    hsv_palette,
	    value_scalars = value_scalars,
	    saturation_scalars = saturation_scalars
	  )
	)

	# middle_palette_index = (middle_palette_index:=len(palettes))//2 + (middle_palette_index & 1)

	if light_palette:
	  bold_colors = palettes[1]
	  base_colors = palettes[-1].copy()
	  base_colors[0] = palettes[-1][0]
	  highlight = bold_colors[highlight_index]
	  lowlight = palettes[-3][highlight_index]

	else:
	  base_colors = palettes[2]
	  bold_colors = palettes[-1]
	  highlight = bold_colors[highlight_index]
	  lowlight = palettes[1][highlight_index]

	# return (base_colors, bold_colors, indices_sorted_by_neighbor_quantity)
	return {
		"base_colors":    base_colors,
		"bold_colors":    bold_colors,
		"highlight":      highlight,
		"lowlight":       lowlight,
		"weighted_order": indices_sorted_by_neighbor_quantity
	}