import React from 'react';
import styled from 'styled-components';
import {
  View,
  ScrollView,
  Image,
  Text,
  Dimensions,
  Alert,
  StyleSheet,
} from 'react-native';
import { TouchableHighlight } from 'react-native-gesture-handler';

const Container = styled.View`
  flex: 1;
  justify-content: center;
  background-color: ${({ theme }) => theme.background};
`;

const StyledText = styled.Text`
  font-size: 30px;
  color: #111111;
`;

var images = [
  require('../assets/a1.png'),
  require('../assets/a2.png'),
  require('../assets/a3.png'),
  require('../assets/a4.png'),
  require('../assets/a5.png'),
  require('../assets/a6.png'),
  require('../assets/a7.png'),
  require('../assets/a8.png'),
  require('../assets/a9.png'),
  require('../assets/a10.png'),
  require('../assets/a11.png'),
  require('../assets/a12.png'),
];

var { width, height } = Dimensions.get('window');

class Mypage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      url: '',
    };
  }

  state = {
    image: null,
  };
  renderSectionOne = () => {
    return images.map((image, index) => {
      return (
        <TouchableHighlight
          //onPress={() => navigation.navigate("Signup")}
          onPress={this.clickImage}
        >
          <View
            key={index}
            style={[
              { width: width / 3 },
              { height: width / 3 },
              index % 3 !== 0 ? { paddingLeft: 2 } : { paddingLeft: 0 },
              { marginBottom: 2 },
            ]}
          >
            <Image
              key={index}
              style={{ flex: 1, width: undefined, height: undefined }}
              source={image}
            />
          </View>
        </TouchableHighlight>
      );
    });
  };

  renderSection = () => {
    return (
      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {this.renderSectionOne()}
      </View>
    );
  };

  clickImage = () =>
    this.props.navigation.navigate('Test', {
      url: this.state.image,
    });

  render() {
    return (
      <Container>
        <Text
          style={{
            fontSize: 25,
            marginTop: 10,
            marginLeft: 20,
            fontWeight: 'bold',
          }}
        >
          Hi, Linist!
        </Text>

        <View style={{ flexDirection: 'row', marginTop: 10 }}>
          <View style={{ flex: 1, alignItems: 'center', marginLeft: 20 }}>
            <Image
              source={require('../assets/2.png')}
              style={{ width: 75, height: 75, borderRadius: 37.5 }}
            />
          </View>
          <View style={{ flex: 3 }}>
            <View
              style={{
                flexDirection: 'row',
                justifyContent: 'space-around',
              }}
            >
              <View style={{ alignItems: 'center' }}>
                <Text
                  style={{ fontSize: 15, color: 'gray', fontWeight: 'bold' }}
                >
                  Posts
                </Text>
                <Text style={{ fontSize: 25 }}>17</Text>
              </View>
              <View style={{ alignItems: 'center' }}>
                <Text
                  style={{ fontSize: 15, color: 'gray', fontWeight: 'bold' }}
                >
                  Follower
                </Text>
                <Text style={{ fontSize: 25 }}>212</Text>
              </View>
            </View>
          </View>
        </View>

        <View style={{ flexDirection: 'row' }}>
          <TouchableHighlight style={styles.button}>
            <View style={styles.btnContainer}>
              <Text
                style={styles.btnText}
                onPress={() => Alert.alert('Edit Profile')}
              >
                Edit Profile
              </Text>
            </View>
          </TouchableHighlight>
          <TouchableHighlight style={styles.button}>
            <View style={styles.btnContainer}>
              <Text
                style={styles.btnText}
                onPress={() => Alert.alert('New Posts')}
              >
                New Posts
              </Text>
            </View>
          </TouchableHighlight>
        </View>

        <ScrollView>{this.renderSection()}</ScrollView>
      </Container>
    );
  }
}

const styles = StyleSheet.create({
  btnContainer: {
    backgroundColor: '#9C9393',
    paddingHorizontal: 50,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 10,
    margin: 7,
  },
  button: {
    borderRadius: 5,
  },
  btnText: {
    textAlign: 'center',
    fontSize: 14,
    color: 'white',
  },
});

export default Mypage;
