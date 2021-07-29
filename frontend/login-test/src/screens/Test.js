import React from 'react';
import { View, Button, Image, StyleSheet } from 'react-native';
export default class Test extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      url: '',
    };
  }
  componentDidMount() {
    this.setState({
      url: this.props.route.params.url,
    });
  }

  render() {
    return (
      <View style={styles.container}>
        <Image
          source={{ uri: this.state.url }}
          style={{ width: '100%', height: '94%' }}
        />
        <Button
          title='Model Translation'
          style={styles.text}
          onPress={() =>
            this.props.navigation.navigate('Background', {
              url: this.state.image,
            })
          }
          color='#EF9DA9'
        />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    height: 600,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
  },

  text1: {
    textAlign: 'left',
    fontWeight: 'bold',
    fontSize: 30,
  },
  text2: {
    textAlign: 'left',
    fontSize: 15,
  },
  btnContainer: {
    backgroundColor: '#EF9DA9',
    paddingHorizontal: 50,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 10,
  },
  button: {
    borderRadius: 5,
  },
  btnText: {
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 16,
    color: 'white',
  },
});
