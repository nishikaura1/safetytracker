import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  CircularProgress,
  ThemeProvider,
  createTheme,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Radio,
  RadioGroup,
  FormControlLabel
} from '@mui/material';
import { GoogleMap, LoadScript, Marker, Polyline, InfoWindow } from '@react-google-maps/api';
import SafetyScore from './components/SafetyScore';
import FeatureContributions from './components/FeatureContributions';
import LAPDAnalysis from './components/LAPDAnalysis';
import axios from 'axios';
import { CheckCircle, Warning, Error as ErrorIcon } from '@mui/icons-material';
import dataRichLocations from './data_rich_locations.json';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const mapContainerStyle = {
  width: '100%',
  height: '400px',
};

const center = {
  lat: 34.0522,
  lng: -118.2437,
};

const DATA_RICH_LOCATIONS = dataRichLocations;

function App() {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [geoMessage, setGeoMessage] = useState("");
  const [selectedDropdown, setSelectedDropdown] = useState("");
  const [fromDropdown, setFromDropdown] = useState("");
  const [toDropdown, setToDropdown] = useState("");
  const [fromLocation, setFromLocation] = useState(null);
  const [toLocation, setToLocation] = useState(null);
  const [locationMode, setLocationMode] = useState('from'); // 'from' or 'to'
  const [showFromInfo, setShowFromInfo] = useState(false);
  const [showToInfo, setShowToInfo] = useState(false);

  const handleLocationModeChange = (event) => {
    setLocationMode(event.target.value);
  };

  const handleMapClick = (event) => {
    const lat = event.latLng.lat();
    const lng = event.latLng.lng();
    if (locationMode === 'from') {
      setFromLocation({ name: 'Custom From', lat, lng });
      setFromDropdown('');
      if (toLocation) fetchRouteSafety({ name: 'Custom From', lat, lng }, toLocation);
    } else {
      setToLocation({ name: 'Custom To', lat, lng });
      setToDropdown('');
      if (fromLocation) fetchRouteSafety(fromLocation, { name: 'Custom To', lat, lng });
    }
  };

  const handleMarkerDragEnd = (type) => (event) => {
    const lat = event.latLng.lat();
    const lng = event.latLng.lng();
    if (type === 'from') {
      setFromLocation({ name: 'Custom From', lat, lng });
      setFromDropdown('');
      if (toLocation) fetchRouteSafety({ name: 'Custom From', lat, lng }, toLocation);
    } else {
      setToLocation({ name: 'Custom To', lat, lng });
      setToDropdown('');
      if (fromLocation) fetchRouteSafety(fromLocation, { name: 'Custom To', lat, lng });
    }
  };

  const handleGetLocation = () => {
    if (navigator.geolocation) {
      setGeoMessage("Getting your location...");
      setLoading(true);
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setGeoMessage("");
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          setSelectedLocation({ lat, lng });
          axios.post('http://localhost:8000/predict', { lat, lon: lng })
            .then((response) => {
              setPrediction(response.data);
              setLoading(false);
            })
            .catch((err) => {
              setError('Error getting safety prediction. Please try again.');
              setLoading(false);
            });
        },
        (error) => {
          setGeoMessage("");
          setError('Unable to retrieve your location.');
          setLoading(false);
        }
      );
    } else {
      setError('Geolocation is not supported by this browser.');
    }
  };

  const handleDropdownChange = (event) => {
    const locationName = event.target.value;
    setSelectedDropdown(locationName);
    const loc = DATA_RICH_LOCATIONS.find(l => l.neighborhood === locationName);
    if (loc) {
      setSelectedLocation({ lat: loc.latitude, lng: loc.longitude });
      setLoading(true);
      axios.post('http://localhost:8000/predict', { lat: loc.latitude, lon: loc.longitude })
        .then((response) => {
          setPrediction(response.data);
          setLoading(false);
        })
        .catch((err) => {
          setError('Error getting safety prediction. Please try again.');
          setLoading(false);
        });
    }
  };

  const handleFromChange = (event) => {
    const neighborhood = event.target.value;
    setFromDropdown(neighborhood);
    const loc = DATA_RICH_LOCATIONS.find(l => l.neighborhood === neighborhood);
    setFromLocation(loc ? { name: loc.neighborhood, lat: loc.latitude, lng: loc.longitude } : null);
  };

  const handleToChange = (event) => {
    const neighborhood = event.target.value;
    setToDropdown(neighborhood);
    const loc = DATA_RICH_LOCATIONS.find(l => l.neighborhood === neighborhood);
    setToLocation(loc ? { name: loc.neighborhood, lat: loc.latitude, lng: loc.longitude } : null);
  };

  const fetchRouteSafety = (from, to) => {
    setLoading(true);
    setSelectedLocation(null); // Clear single marker
    axios.get('http://localhost:8000/route-safety', {
      params: {
        home_lat: from.lat,
        home_lon: from.lng,
        school_lat: to.lat,
        school_lon: to.lng
      }
    })
      .then((response) => {
        setPrediction(response.data.summary || response.data);
        setLoading(false);
      })
      .catch((err) => {
        setError('Error getting route safety prediction. Please try again.');
        setLoading(false);
      });
  };

  const handleCalculate = () => {
    if (fromLocation && toLocation) {
      fetchRouteSafety(fromLocation, toLocation);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Card elevation={6} sx={{ mb: 4, p: 2, maxWidth: 600, mx: 'auto', background: '#fafbfc' }}>
            <CardContent>
              <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 700, color: '#1976d2' }}>
                Safety Tracker
              </Typography>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="from-location-label">From (Neighborhood)</InputLabel>
                <Select
                  labelId="from-location-label"
                  value={fromDropdown}
                  label="From (Neighborhood)"
                  onChange={handleFromChange}
                >
                  <MenuItem value=""><em>None</em></MenuItem>
                  {DATA_RICH_LOCATIONS.map((loc) => (
                    <MenuItem key={loc.neighborhood} value={loc.neighborhood}>{loc.neighborhood}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="to-location-label">To (Neighborhood)</InputLabel>
                <Select
                  labelId="to-location-label"
                  value={toDropdown}
                  label="To (Neighborhood)"
                  onChange={handleToChange}
                >
                  <MenuItem value=""><em>None</em></MenuItem>
                  {DATA_RICH_LOCATIONS.map((loc) => (
                    <MenuItem key={loc.neighborhood} value={loc.neighborhood}>{loc.neighborhood}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button
                variant="contained"
                color="primary"
                onClick={handleGetLocation}
                sx={{ mb: 2, width: '100%' }}
              >
                USE MY LOCATION
              </Button>
              {geoMessage && (
                <Typography align="center" color="text.secondary" sx={{ mb: 2 }}>
                  {geoMessage}
                </Typography>
              )}
              {error && (
                <Box display="flex" alignItems="center" justifyContent="center" sx={{ mt: 2 }}>
                  <ErrorIcon color="error" sx={{ mr: 1 }} />
                  <Typography color="error" align="center" gutterBottom>
                    {error}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>

          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
            <FormControl component="fieldset">
              <RadioGroup row value={locationMode} onChange={handleLocationModeChange}>
                <FormControlLabel value="from" control={<Radio />} label="Set From" />
                <FormControlLabel value="to" control={<Radio />} label="Set To" />
              </RadioGroup>
            </FormControl>
          </Box>

          <Button
            variant="contained"
            color="secondary"
            onClick={handleCalculate}
            disabled={!(fromLocation && toLocation)}
            sx={{ mb: 2, width: '100%' }}
          >
            Calculate Safety Prediction
          </Button>

          <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
            <LoadScript googleMapsApiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY}>
              <GoogleMap
                mapContainerStyle={mapContainerStyle}
                center={fromLocation || center}
                zoom={12}
                onClick={handleMapClick}
              >
                {selectedLocation && (
                  <Marker
                    position={selectedLocation}
                    animation={window.google.maps.Animation.DROP}
                  />
                )}
                {fromLocation && (
                  <Marker
                    position={{ lat: fromLocation.lat, lng: fromLocation.lng }}
                    label="A"
                    icon={{ url: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png' }}
                    draggable
                    onDragEnd={handleMarkerDragEnd('from')}
                    onClick={() => setShowFromInfo(true)}
                  />
                )}
                {showFromInfo && fromLocation && (
                  <InfoWindow
                    position={{ lat: fromLocation.lat, lng: fromLocation.lng }}
                    onCloseClick={() => setShowFromInfo(false)}
                  >
                    <div>
                      <Typography variant="subtitle2">{fromLocation.name || 'From'}</Typography>
                      <Typography variant="caption">{fromLocation.lat.toFixed(5)}, {fromLocation.lng.toFixed(5)}</Typography>
                    </div>
                  </InfoWindow>
                )}
                {toLocation && (
                  <Marker
                    position={{ lat: toLocation.lat, lng: toLocation.lng }}
                    label="B"
                    icon={{ url: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png' }}
                    draggable
                    onDragEnd={handleMarkerDragEnd('to')}
                    onClick={() => setShowToInfo(true)}
                  />
                )}
                {showToInfo && toLocation && (
                  <InfoWindow
                    position={{ lat: toLocation.lat, lng: toLocation.lng }}
                    onCloseClick={() => setShowToInfo(false)}
                  >
                    <div>
                      <Typography variant="subtitle2">{toLocation.name || 'To'}</Typography>
                      <Typography variant="caption">{toLocation.lat.toFixed(5)}, {toLocation.lng.toFixed(5)}</Typography>
                    </div>
                  </InfoWindow>
                )}
                {fromLocation && toLocation && (
                  <Polyline
                    path={[
                      { lat: fromLocation.lat, lng: fromLocation.lng },
                      { lat: toLocation.lat, lng: toLocation.lng }
                    ]}
                    options={{ strokeColor: '#1976d2', strokeWeight: 4, strokeOpacity: 0.7 }}
                  />
                )}
              </GoogleMap>
            </LoadScript>
          </Paper>

          {loading && (
            <Box display="flex" justifyContent="center" my={3}>
              <CircularProgress />
            </Box>
          )}

          {prediction && !loading && (
            <>
              <Box sx={{ mb: 3 }}>
                <Paper elevation={4} sx={{ p: 3, textAlign: 'center', background: '#f5f5f5' }}>
                  <Box display="flex" alignItems="center" justifyContent="center" sx={{ mb: 1 }}>
                    {(() => {
                      const score = prediction.adjusted_score || prediction.base_score;
                      if (score >= 0.8) return <CheckCircle sx={{ color: '#4caf50', fontSize: 40, mr: 1 }} />;
                      if (score >= 0.6) return <Warning sx={{ color: '#ff9800', fontSize: 40, mr: 1 }} />;
                      return <ErrorIcon sx={{ color: '#f44336', fontSize: 40, mr: 1 }} />;
                    })()}
                    <Typography variant="h5" gutterBottom>
                      Safety Prediction
                    </Typography>
                  </Box>
                  <Typography variant="subtitle1" color="text.secondary">
                    Location: {fromLocation && toLocation ? `${fromLocation.name} → ${toLocation.name}` : selectedLocation ? `${selectedLocation.lat.toFixed(5)}, ${selectedLocation.lng.toFixed(5)}` : 'N/A'}
                  </Typography>
                  <Typography variant="subtitle2" color="text.secondary">
                    Time: {prediction.timestamp ? new Date(prediction.timestamp).toLocaleString() : 'N/A'}
                  </Typography>
                  <Typography variant="h6" sx={{ mt: 1, color: (() => {
                    const score = prediction.adjusted_score || prediction.base_score;
                    if (score >= 0.8) return '#4caf50';
                    if (score >= 0.6) return '#ff9800';
                    return '#f44336';
                  })() }}>
                    Safety Score: {typeof (prediction.adjusted_score || prediction.base_score) === 'number' ? Math.round((prediction.adjusted_score || prediction.base_score) * 100) : 'N/A'}
                  </Typography>
                </Paper>
              </Box>
              <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
                <SafetyScore score={prediction.adjusted_score || prediction.base_score} />
                <FeatureContributions contributions={prediction.feature_contributions} />
                {prediction.lapd_analysis && (
                  <LAPDAnalysis analysis={prediction.lapd_analysis} />
                )}
              </Box>
            </>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App; 