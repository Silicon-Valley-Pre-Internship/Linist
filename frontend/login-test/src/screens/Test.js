import React from 'react';
import styled from 'styled-components';
import { View, Button, Image, StyleSheet, Text } from 'react-native';

const Container = styled.View`
  flex: 1;
  justify-content: center;
  align-items: center;
  background-color: ${({ theme }) => theme.background};
  padding: 0 20px;
`;

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
      <Container>
        <View style={styles.container}>
          <Text style={styles.text1}>Transformed Image!</Text>
          <Text style={styles.text1}></Text>

          <Image
            source={this.state.url ? { uri: this.state.url } : null}
            style={{ width: '100%', height: '90%' }}
          />
          {/* <Button
          title='Model Translation'
          style={styles.text}
          onPress={() =>
            this.props.navigation.navigate('Background', {
              url: this.state.image,
            })
          }
          color='#EF9DA9'
        /> */}
        </View>
      </Container>
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
