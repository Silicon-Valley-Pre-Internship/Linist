import React, { Component } from 'react';
import { View, Text, StyleSheet } from 'react-native';

import { Container, Content, Icon } from 'native-base';
import CardComponent from '../components/CardComponent';

class Instagram extends Component {
  render() {
    return (
      <Container style={styles.container}>
        <Content>
          <CardComponent />
        </Content>
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
