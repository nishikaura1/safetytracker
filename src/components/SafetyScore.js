import React from 'react';
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const ScoreContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  textAlign: 'center',
}));

const ScoreCircle = styled(Box)(({ theme }) => ({
  position: 'relative',
  display: 'inline-flex',
  margin: theme.spacing(2, 0),
}));

const ScoreValue = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  bottom: 0,
  right: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const getScoreColor = (score) => {
  if (score >= 80) return '#4caf50'; // Green
  if (score >= 60) return '#ff9800'; // Orange
  return '#f44336'; // Red
};

const SafetyScore = ({ score }) => {
  const normalizedScore = Math.min(Math.max(score, 0), 100);
  const color = getScoreColor(normalizedScore);

  return (
    <ScoreContainer elevation={3}>
      <Typography variant="h6" gutterBottom>
        Safety Score
      </Typography>
      <ScoreCircle>
        <CircularProgress
          variant="determinate"
          value={normalizedScore}
          size={120}
          thickness={4}
          sx={{ color }}
        />
        <ScoreValue>
          <Typography variant="h4" component="div" color="text.secondary">
            {Math.round(normalizedScore)}
          </Typography>
        </ScoreValue>
      </ScoreCircle>
      <Typography variant="body1" color="text.secondary">
        {normalizedScore >= 80 ? 'Very Safe' :
         normalizedScore >= 60 ? 'Moderately Safe' :
         'Exercise Caution'}
      </Typography>
    </ScoreContainer>
  );
};

export default SafetyScore; 