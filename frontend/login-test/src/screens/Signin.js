import React, { useContext, useState, useRef, useEffect } from 'react';
import { ThemeContext } from 'styled-components/native';
import styled from 'styled-components';
import { View, Button, Text, StyleSheet, ImageBackground } from 'react-native';
import { Image, Input, ErrorMessage } from '../components';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { KeyboardAwareScrollView } from 'react-native-keyboard-aware-scroll-view';
import { signin } from '../firebase';
import { Alert } from 'react-native';
import { TouchableHighlight } from 'react-native-gesture-handler';
import { validateEmail, removeWhitespace } from '../utils';
import { UserContext, ProgressContext } from '../contexts';
const Container = styled.View`
  justify-content: center;
  align-items: flex-start;
  background-color: ${({ theme }) => theme.background};
  padding: 0px 20px;
  padding-top: ${({ insets: { top } }) => top}px;
  padding-bottom: ${({ insets: { bottom } }) => bottom}px;
`;
const LOGO =
  'https://firebasestorage.googleapis.com/v0/b/linist-test.appspot.com/o/linist_logo.png?alt=media';
const Signin = ({ navigation }) => {
  const insets = useSafeAreaInsets();
  const theme = useContext(ThemeContext);
  const { setUser } = useContext(UserContext);
  const { spinner } = useContext(ProgressContext); //로그인을 시도하는 동안 spinner
  //useState를 이용해서 email, password 상태 변수를 만든다
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [disabled, setDisabled] = useState(true);
  const refPassword = useRef(null);
  useEffect(() => {
    setDisabled(!(email && password && !errorMessage));
  }, [email, password, errorMessage]);
  const _handleEmailChange = (email) => {
    const changedEmail = removeWhitespace(email);
    setEmail(changedEmail);
    setErrorMessage(
      validateEmail(changedEmail) ? '' : 'Please verify your email'
    );
  };
  const _handlePasswordChange = (password) => {
    setPassword(removeWhitespace(password));
  };
  const _handleSigninBtnPress = async () => {
    try {
      spinner.start();
      const user = await signin({ email, password });
      setUser(user);
    } catch (e) {
      Alert.alert('Signin Error', e.message);
    } finally {
      spinner.stop(); //로그인 성공 여부와 상관없이 spinner 사라져야 함
    }
  };
  return (
    <KeyboardAwareScrollView
      extraScrollHeight={20}
      contentContainerStyle={{ flex: 1 }}
    >
      <Container insets={insets}>
        <ImageBackground
          style={{ width: '100%', height: '100%' }}
          source={require('../assets/login_bgr.png')}
          resizeMode='stretch'
        >
          {/* <Text style={styles.text1}></Text>
          <Text style={styles.text1}></Text> */}
          <Text style={styles.text1}></Text>
          <Text style={styles.text1}></Text>
          <Image url={LOGO} />
          <Text style={styles.text1}>Welcome User</Text>
          <Text style={styles.text2}>Sign in to continue</Text>
          <Text style={styles.text1}></Text>
          <Input
            label='Email'
            placeholder='Email'
            returnKeyType='next'
            value={email}
            onChangeText={_handleEmailChange}
            onSubmitEditing={() => refPassword.current.focus()}
          />
          <Input
            ref={refPassword}
            label='Password'
            placeholder='Password'
            returnKeyType='done'
            value={password}
            onChangeText={_handlePasswordChange}
            isPassword={true}
            onSubmitEditing={_handleSigninBtnPress}
          />
          <ErrorMessage message={errorMessage} />
          <TouchableHighlight
            onPress={_handleSigninBtnPress}
            style={styles.button}
          >
            <View style={styles.btnContainer}>
              <Text style={styles.btnText}>SIGN IN</Text>
            </View>
          </TouchableHighlight>
          <Text></Text>
          <TouchableHighlight
            onPress={() => navigation.navigate('Signup')}
            style={styles.button}
          >
            <View style={styles.btnContainer}>
              <Text style={styles.btnText}>SIGN UP</Text>
            </View>
          </TouchableHighlight>
        </ImageBackground>
      </Container>
    </KeyboardAwareScrollView>
  );
};
const styles = StyleSheet.create({
  container2: {
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
  imagebg: {
    width: '100%',
    height: '100%',
  },
  text2: {
    textAlign: 'left',
    color: '#C7C6C1',
    fontSize: 25,
    fontWeight: 'bold',
  },
  btnContainer: {
    justifyContent: 'center',
    backgroundColor: '#F2B8C6',
    paddingHorizontal: 50,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    borderRadius: 8,
    justifyContent: 'center',
    width: 330,
  },
  button: {
    borderRadius: 5,
  },
  btnText: {
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 17,
    color: 'white',
  },
});
export default Signin;
