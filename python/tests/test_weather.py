"""The shared scene weather (tp.Weather, tp.RainCell): one description that the
ocean and the marine radar both read. No renderer: constructing a MarineRadar
traces nothing until update(renderer) is called.
"""
import math

import pytest

import threepp as tp


def test_fields_round_trip():
    w = tp.Weather()
    w.wind_speed = 7.5
    w.wind_theta = 1.25
    w.fetch = 12000.0
    w.swell_height = 0.75
    w.swell_period = 9.0
    w.swell_theta = -0.5
    w.rain = [tp.RainCell(100.0, -200.0, radius=150.0, top=1400.0, rain_rate=16.0, vx=2.0, vz=-3.0)]
    assert w.wind_speed == pytest.approx(7.5)
    assert w.wind_theta == pytest.approx(1.25)
    assert w.fetch == pytest.approx(12000.0)
    assert w.swell_height == pytest.approx(0.75)
    assert w.swell_period == pytest.approx(9.0)
    assert w.swell_theta == pytest.approx(-0.5)
    assert len(w.rain) == 1
    cell = w.rain[0]
    assert (cell.centre.x, cell.centre.z) == pytest.approx((100.0, -200.0))
    assert (cell.velocity.x, cell.velocity.z) == pytest.approx((2.0, -3.0))
    assert cell.radius == pytest.approx(150.0)
    assert cell.top == pytest.approx(1400.0)
    assert cell.rain_rate == pytest.approx(16.0)
    # The rain rate at a moving cell's centre, where its velocity has carried it.
    assert w.rain_rate_at(100.0 + 2.0 * 10.0, -200.0 - 3.0 * 10.0, 10.0) == pytest.approx(16.0)


def test_rain_is_a_copy():
    w = tp.Weather()
    w.rain = [tp.RainCell(0.0, 0.0)]
    w.rain[0].rain_rate = 99.0  # changes the copy only
    assert w.rain[0].rain_rate == pytest.approx(8.0)


def test_significant_wave_height_fully_developed():
    w = tp.Weather()
    w.wind_speed = 10.0
    w.fetch = 0.0
    assert w.significant_wave_height() == pytest.approx(0.21 * 100.0 / 9.81, rel=1e-4)
    assert w.significant_wave_height() == pytest.approx(2.141, rel=0.03)


@pytest.mark.skipif(not getattr(tp, "HAS_VULKAN", False), reason="MarineRadar needs the Vulkan build")
def test_radar_shares_the_weather_object():
    radar = tp.MarineRadar()
    w = tp.Weather()
    radar.weather = w
    w.wind_speed = 3.25
    assert radar.weather.wind_speed == pytest.approx(3.25)
    radar.weather.fetch = 4000.0
    assert w.fetch == pytest.approx(4000.0)


def test_model_has_no_sea_state():
    m = tp.MarineRadarModel()
    assert not hasattr(m, "sea_state")
    assert not hasattr(m, "rain")


def test_radar_helpers():
    lam = 299792458.0 / 9.41e9
    assert 10.0 * math.log10(tp.rain_volume_reflectivity(4.0, lam)) == pytest.approx(-63.0, abs=0.2)
    assert tp.rain_specific_attenuation(16.0) == pytest.approx(0.347, abs=0.005)
    for s in (0.0, 1.0, 2.3, 6.0):
        assert tp.sea_state_from_wave_height(tp.douglas_wave_height(s)) == pytest.approx(s)
