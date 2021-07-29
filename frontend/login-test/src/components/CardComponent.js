import React, { Component } from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';

import {
  Card,
  CardItem,
  Thumbnail,
  Body,
  Left,
  Right,
  Button,
  Icon,
} from 'native-base';

class CardComponent extends Component {
  render() {
    const image = require('../assets/feed_images/1.jpg');

    return (
      <Card>
        <CardItem>
          <Left>
            <Thumbnail source={require('../assets/me.png')} />
            <Body>
              <Text>Yellme </Text>
              <Text note>July 29, 2021</Text>
            </Body>
          </Left>
        </CardItem>
        <CardItem cardBody>
          <Image source={image} style={{ height: 200, width: null, flex: 1 }} />
        </CardItem>
        <CardItem style={{ height: 45 }}>
          <Left>
            <Button transparent>
              <Icon name='ios-heart-outline' style={{ color: 'black' }} />
            </Button>
            <Button transparent>
              <Icon name='ios-chatbubbles-outline' style={{ color: 'black' }} />
            </Button>
            <Button transparent>
              <Icon name='ios-send-outline' style={{ color: 'black' }} />
            </Button>
          </Left>
        </CardItem>

        <CardItem style={{ height: 20 }}>
          <Text> 101 </Text>
        </CardItem>
        <CardItem>
          <Body>
            <Text>
              <Text style={{ fontWeight: '900' }}>yellme</Text>
              card explanation
            </Text>
          </Body>
        </CardItem>
      </Card>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

export default CardComponent;
