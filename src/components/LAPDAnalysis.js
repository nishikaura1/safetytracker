import React from 'react';
import { Paper, Typography, Box, Grid, Chip } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const LAPDAnalysis = ({ analysis }) => {
  const crimeTypeData = [
    { name: 'Violent', value: analysis.crime_distribution.violent },
    { name: 'Property', value: analysis.crime_distribution.property },
    { name: 'Quality of Life', value: analysis.crime_distribution.quality_of_life }
  ];

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        LAPD Crime Analysis
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={crimeTypeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {crimeTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>
              Key Metrics
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                label={`Total Crimes: ${analysis.total_crimes}`}
                color="primary"
                variant="outlined"
              />
              <Chip 
                label={`Recent Crimes: ${analysis.recent_crimes}`}
                color="secondary"
                variant="outlined"
              />
              <Chip 
                label={`Crime Density: ${analysis.crime_density.toFixed(2)}`}
                color="info"
                variant="outlined"
              />
              <Chip 
                label={`Night Risk: ${analysis.night_risk.toFixed(2)}`}
                color="warning"
                variant="outlined"
              />
            </Box>
          </Box>
          
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Risk Factors
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Typography variant="body2">
                • Temporal Risk: {analysis.temporal_risk.toFixed(2)}
              </Typography>
              <Typography variant="body2">
                • Spatial Risk: {analysis.spatial_risk.toFixed(2)}
              </Typography>
              <Typography variant="body2">
                • Overall Risk Score: {analysis.overall_risk_score.toFixed(2)}
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default LAPDAnalysis; 