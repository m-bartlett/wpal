- better config
  - fails on missing section
  - maybe just iterate .sections()?
  - how to handle wallpaper-picker-default and wallpaper-path-command?

- different minimum contrast for light and dark, also in config.ini

- iterative contrast increasor uses HSL instead of contrast gradient descent

- exif load skips kmeans execution without making code ugly
  - main -> core.main() and then __main__ imports core.main ?

- fix verbose color printing now that we don't generate 5 palettes

- ~~main -> __main__ and update symlink~~
- ~~exif only stores base and bold colors, re-derives 5-palette~~ this loses information about cluster-sorting of colors