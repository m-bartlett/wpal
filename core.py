from image import *
from kmeans import kmeans


def generate_ANSI_palette_from_pixels( *,
                                       rgb_pixels,
                                       kmeans_initial_colors,
                                       kmeans_iterations = 3,
                                       minimum_contrast = 0,
                                       light_palette = False,
                                       overwrites = {},
                                       value = 1.0,
                                       value_delta = 0.5,  # value as in HSV
                                       base_saturation = 1.0,
                                       bold_saturation = 1.1,
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
	# else:
		# initial_palette[0] += 0x44
		# initial_palette[0] /= 3
		# initial_palette[0]*=0
		# initial_palette[0]+=0x25


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



	if light_palette:
		# swap base and bold for light themes so bold is still high contrast
	  # base_value, bold_value = bold_value, base_value
	  # base_saturation, bold_saturation = bold_saturation, base_saturation
	  value_delta_2 = value_delta / 2
	  base_value = value + value_delta_2
	  bold_value = value - value_delta_2
	  bold_saturation += 0.25
	else:
	  base_value = value
	  bold_value = value + value_delta

	hsv_palette = rgb_palette_to_hsv_palette(initial_palette)
	base_colors = rebalance_palette(hsv_palette, base_value, base_saturation)
	bold_colors = rebalance_palette(hsv_palette, bold_value, bold_saturation)

	for color_index in range(0,8):
		if (color := overwrites.get(color_index)):
			base_colors[color_index] = hex2rgb(color)

	for color_index in range(8,16):
		if (color := overwrites.get(color_index)):
			bold_colors[color_index&7] = hex2rgb(color)

	highlight_index = get_most_saturated_color_index(hsv_palette)
	highlight = bold_colors[highlight_index]
	lowlight  = base_colors[highlight_index]



	# return (base_colors, bold_colors, indices_sorted_by_neighbor_quantity)
	return {
		"base_colors":    base_colors,
		"bold_colors":    bold_colors,
		"highlight":      highlight,
		"lowlight":       lowlight,
		"weighted_order": indices_sorted_by_neighbor_quantity
	}