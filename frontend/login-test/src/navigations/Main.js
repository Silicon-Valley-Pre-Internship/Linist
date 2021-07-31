import React, { useContext } from 'react';
import { ThemeContext } from 'styled-components/native';
import { createStackNavigator } from '@react-navigation/stack';
import { MainScreen, Profile, Test } from '../screens';
import { Auth } from './Auth';
const Stack = createStackNavigator();

const Main = () => {
  const theme = useContext(ThemeContext);

  return (
    <Stack.Navigator
      screenOptions={{
        headerTitleAlign: 'center',
        headerTintColor: theme.text,
        headerBackTitleVisible: false,
        cardStyle: { backgroundColor: theme.background },
      }}
    >
      <Stack.Screen name='Welcome Linist' component={MainTab} />
      <Stack.Screen name='Profile' component={Profile} />
      <Stack.Screen name='MainScreen' component={MainScreen} />
      <Stack.Screen name='Result Image Page' component={Test} />
    </Stack.Navigator>
  );
};

export default Main;