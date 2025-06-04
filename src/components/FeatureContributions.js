import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const FeatureContributions = ({ contributions }) => {
  // Transform contributions object into array format for Recharts
  const data = Object.entries(contributions || {}).map(([name, value]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: Math.abs(value),
    impact: value > 0 ? 'positive' : 'negative'
  })).sort((a, b) => b.value - a.value);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Feature Contributions
      </Typography>
      <Box sx={{ height: 300, width: '100%' }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              angle={-45}
              textAnchor="end"
              height={100}
              interval={0}
              tick={{ fontSize: 12 }}
            />
            <YAxis />
            <Tooltip 
              formatter={(value, name, props) => [
                `${value.toFixed(2)}`,
                'Contribution'
              ]}
            />
            <Bar 
              dataKey="value" 
              fill="#8884d8"
              fillOpacity={0.8}
            />
          </BarChart>
        </ResponsiveContainer>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        Shows how each factor contributes to the safety score
      </Typography>
    </Paper>
  );
};

export default FeatureContributions; 