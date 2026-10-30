# Dataset location

The original CSV is preserved at the workspace root as
`04_smart_building_energy_digital_twin2.csv`. The supplied workspace did not
contain a file named `04_smart_building_energy_digital_twin.csv`.

The backend checks both requested root filenames and supports an explicit path
through the `BUILDING_DATASET` environment variable. No copy of the input CSV
is made here.
