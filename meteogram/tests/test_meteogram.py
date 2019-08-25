"""Test use of the meteogram module."""

from unittest.mock import patch
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import os
from datetime import datetime
import numpy as np
from pandas import DataFrame
from matplotlib.pyplot import figure
from meteogram import meteogram


@pytest.fixture
def load_example_asos() -> DataFrame:
    """
    Fixture to load example data
    """
    example_data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "staticdata")
    )

    data_path = os.path.join(example_data_path, "AMW_example_data.csv")
    return meteogram.download_asos_data(data_path)


# Example starter test
def test_degF_to_degC_at_freezing() -> None:
    """Test if celsius conversion is correct at freezing.
    """
    # Setup
    freezing_degF = 32.0
    freezing_degC = 0.0

    # Exercise
    result = meteogram.degF_to_degC(freezing_degF)

    # Verify
    assert result == freezing_degC

    # Cleanup - none necessary


# Instructor led introductory examples
def test_title_case() -> None:
    """Test whether title case conversion works
    """
    # Setup
    in_string = "This is a string"
    desired = "This Is A String"

    # Exercise
    result = desired.title()

    # Verify
    assert result == desired

    # Cleanup - none necessary


# Exercise 1
def test_build_asos_request_url_single_digit_datetimes():
    """
    Test building URL with single digit month and day.
    """
    # Setup
    start = datetime(2018, 1, 5, 1)
    end = datetime(2018, 1, 9, 1)
    station = "FSD"

    # Exercise
    result_url = meteogram.build_asos_request_url(
        station=station, start_date=start, end_date=end
    )

    # Verify
    desired_url = "https://mesonet.agron.iastate.edu/request/asos/1min_dl.php?station%5B%5D=FSD&tz=UTC&year1=2018&month1=01&day1=05&hour1=01&minute1=00&year2=2018&month2=01&day2=09&hour2=01&minute2=00&vars%5B%5D=tmpf&vars%5B%5D=dwpf&vars%5B%5D=sknt&vars%5B%5D=drct&sample=1min&what=view&delim=comma&gis=yes"
    assert result_url == desired_url

    # Cleanup -- none needed


def test_build_asos_request_url_double_digit_datetimes() -> None:
    """
    Test building URL with double digit month and day.
    """
    # Setup
    start = datetime(2018, 10, 11, 11)
    end = datetime(2018, 10, 16, 11)
    station = "FSD"

    # Exercise
    result_url = meteogram.build_asos_request_url(
        station=station, start_date=start, end_date=end
    )

    # Verify
    desired_url = "https://mesonet.agron.iastate.edu/request/asos/1min_dl.php?station%5B%5D=FSD&tz=UTC&year1=2018&month1=10&day1=11&hour1=11&minute1=00&year2=2018&month2=10&day2=16&hour2=11&minute2=00&vars%5B%5D=tmpf&vars%5B%5D=dwpf&vars%5B%5D=sknt&vars%5B%5D=drct&sample=1min&what=view&delim=comma&gis=yes"
    assert result_url == desired_url

    # Cleanup -- none needed


# Exercise 1 - Stop Here


def test_does_three_equal_three() -> None:
    assert 3 == 3


def test_floating_point_subtraction() -> None:
    # Setup
    desired = 0.293
    tolerance = 0.0001

    # Exercise
    actual = 1 - 0.707

    # Verify
    assert_almost_equal(actual, desired)

    # Cleanup -- none needed


# Exercise 2 - Add calculation tests here


def test_wind_components_north() -> None:
    """Ensure correct results when testing wind coming from 0 degrees
    """
    # Setup
    theta = 0
    speed = 1

    # Exercise
    u_calc, v_calc = meteogram.convert_to_wind_components(theta=theta, speed=speed)

    # Verify
    u_desired = 0
    v_desired = -1
    assert_almost_equal(u_calc, u_desired)
    assert_almost_equal(v_calc, v_desired)

    # Cleanup -- none needed


def test_wind_components_northeast() -> None:
    """Ensure correct results when testing wind speed of 45 degrees
    """
    # Setup
    theta = 45
    speed = 10

    # Exercise
    u_calc, v_calc = meteogram.convert_to_wind_components(theta=theta, speed=speed)

    # Verify
    u_desired = -7.0710  # could use np.sqrt here but prefer to be exact
    v_desired = -7.0710
    assert_almost_equal(u_calc, u_desired, 4)
    assert_almost_equal(v_calc, v_desired, 4)


def test_wind_components_north_360() -> None:
    """Ensure correct results when testing wind speed of 360 degrees
    """
    # Setup
    theta = 360
    speed = 1

    # Exercise
    u_calc, v_calc = meteogram.convert_to_wind_components(theta=theta, speed=speed)

    # Verify
    u_desired = 0
    v_desired = -1
    assert_almost_equal(u_calc, u_desired)
    assert_almost_equal(v_calc, v_desired)

    # Cleanup -- none needed


def test_wind_components_no_wind() -> None:
    """Ensure correct results when testing wind speed of 45 degrees
    """
    # Setup
    theta = 45
    speed = 0

    # Exercise
    u_calc, v_calc = meteogram.convert_to_wind_components(theta=theta, speed=speed)

    # Verify
    u_desired = 0
    v_desired = 0
    assert_almost_equal(u_calc, u_desired)
    assert_almost_equal(v_calc, v_desired)


def test_wind_components() -> None:
    """Test wind components in a loop
    
    This makes all the previous tests irrelevant, and still isn't great
    """
    # Setup
    speed = np.array([10, 10, 10, 0])
    direction = np.array([0, 45, 360, 45])

    # Exercise
    u, v = meteogram.convert_to_wind_components(theta=direction, speed=speed)

    # Verify
    true_u = np.array([0, -7.0701, 0, 0])
    true_v = np.array([-10, -7.0701, -10, 0])
    assert_array_almost_equal(u, true_u, 3)
    assert_array_almost_equal(v, true_v, 3)

    # Cleanup -- none needed


#
# Instructor led mock example
#
def mocked_current_utc_time() -> datetime:
    """Mock the UTC time for testing with defaults
    """
    return datetime(2018, 3, 26, 12)


@patch("meteogram.meteogram.current_utc_time", new=mocked_current_utc_time)
def test_mocked_current_utc_time() -> None:
    """Test if we really got the mock set up correctly
    """
    # Setup - None

    # Exercise
    results = meteogram.current_utc_time()

    # Verify
    desired = datetime(2018, 3, 26, 12)
    assert results == desired

    # Cleanup -- none needed


#
# Exercise 3
#
@patch("meteogram.meteogram.current_utc_time", new=mocked_current_utc_time)
def test_build_asos_request_url_defaults() -> None:
    """Make sure that we get the correct URL when we are using defaults
    """
    # Setup - None
    station = "FSD"

    # Exercise
    url = meteogram.build_asos_request_url(station=station)

    # Verify
    desired = "https://mesonet.agron.iastate.edu/request/asos/1min_dl.php?station%5B%5D=FSD&tz=UTC&year1=2018&month1=03&day1=25&hour1=12&minute1=00&year2=2018&month2=03&day2=26&hour2=12&minute2=00&vars%5B%5D=tmpf&vars%5B%5D=dwpf&vars%5B%5D=sknt&vars%5B%5D=drct&sample=1min&what=view&delim=comma&gis=yes"
    assert url == desired

    # Cleanup - None


# Exercise 4 - Add any tests that you can to increase the library coverage.
# think of cases that may not change coverage, but should be tested
# for as well.


def test_current_utc_time() -> None:
    """Ensure that the UTC time returned is valid and is larger than the
    UTC time at time of writing this test
    """
    # Setup - None

    # Exercise
    result = meteogram.current_utc_time()

    # Verify
    desired = datetime.utcnow()
    delta = desired - result
    assert delta.microseconds < 1000

    # Cleanup - None needed


def test_potential_temperature() -> None:
    """Test a potential temperature calculation
    """
    # Setup
    pressure = 800
    temperature = 273
    # Exercise
    potential_temperature = meteogram.potential_temperature(
        pressure=pressure, temperature=temperature
    )
    # Verify
    desired = 290.96
    assert_almost_equal(potential_temperature, desired, 2)
    # Cleanup - None

def test_exner_function() -> None:
    """Test exner function calculation.
    """
    # Setup - none necessary

    # Exercise
    result = meteogram.exner_function(500)

    # Verify
    truth = 0.8203833
    assert_almost_equal(result, truth, 4)

    # Cleanup - none necessary
    
#
# Exercise 4 - Stop Here
#

#
# Instructor led example of image testing
#


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plotting_meteogram_defaults(load_example_asos):
    """Test default meteogram plotting."""
    # Setup
    df = load_example_asos
    # Exercise
    fig, _, _, _ = meteogram.plot_meteogram(df)
    # Verify - Done by decorator when run with -mpl flag
    # Cleanup - none necessary
    return fig


#
# Exercise 5
#

#
# Exercise 5 - Stop Here
#

#
# Exercise 6
#

#
# Exercise 6 - Stop Here
#

#
# Exercise 7
#

#
# Exercise 7 - Stop Here
#

# Demonstration of TDD here (time permitting)
