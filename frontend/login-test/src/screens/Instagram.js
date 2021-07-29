import React, { Component } from 'react';
import styled from 'styled-components';
import {
  View,
  Text,
  StyleSheet,
  ImageBackground,
  Image,
  ScrollView,
} from 'react-native';
import { Content, Icon } from 'native-base';
import CardComponent from '../components/CardComponent';

const Container = styled.View`
  flex: 1;
  justify-content: center;
  align-items: center;
  background-color: ${({ theme }) => theme.background};
  padding: 0 20px;
`;

class Instagram extends React.Component {
  render() {
    return (
      <Container>
        {/* <CardComponent /> */}
        {/* <ImageBackground
          style={{ width: '100%', height: '100%' }}
          source={require('../assets/instagram.jpg')}
          resizeMode='stretch'
        ></ImageBackground> */}

        <Image
          style={{ width: '110%', height: '100%' }}
          source={require('../assets/instagram.jpg')}
        />
      </Container>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'white',
  },
});

export default Instagram;
