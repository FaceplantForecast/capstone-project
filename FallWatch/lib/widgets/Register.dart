import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class RegisterForm extends StatefulWidget {
  const RegisterForm({super.key});

  @override
  State<RegisterForm> createState() => _RegisterFormState();
}

class _RegisterFormState extends State<RegisterForm> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmController = TextEditingController();
  final _usernameController = TextEditingController();
  
  bool _isPasswordVisible = false;
  bool showUsernameField = false;
  bool _isLoading = false;

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmController.dispose();
    _usernameController.dispose();
    super.dispose();
  }

  Future<void> _submitForm() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final name = _nameController.text.trim();
      final email = _emailController.text.trim();
      final password = _passwordController.text.trim();

      final userCredential = 
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: email,
          password: password,
      );
      
      final user = userCredential.user!;

      await user.sendEmailVerification();

      await FirebaseFirestore.instance
        .collection('users')
        .doc(user.uid)
        .set({
          'name' : name,
          'email' : email,
          'username' : _usernameController.text.trim().isNotEmpty
            ? _usernameController.text.trim() : email.split('@')[0],
          'role' : 'caretaker',
          'created_at' : FieldValue.serverTimestamp(),
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Registered successfully, please verify your email.')),
      ); 
      Navigator.pop(context);
    } on FirebaseException catch (e) {
      String regFail = 'Registration failed';

      if (e.code == 'email-already-in-use') {
        regFail = 'User with this email already exists';
      } else if (e.code == 'weak-password') {
        regFail = 'Password is too weak';
      } else if (e.code == 'invalid-email') {
        regFail = 'Email address is invalid';
      }
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar
              (content: Text(regFail),
            ),
          );
      } finally {
        setState(() => _isLoading = false);
      }
    }
  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Card(
            elevation: 10,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
            child: Padding(
              padding: const EdgeInsets.all(28),
              child: Form(
                key: _formKey,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [

                    Image.asset('assets/fallwatch_nb.png', width: 125, height: 125),
                    const SizedBox(height: 5),

                    const Text(
                      "Create an Account",
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      "Sign up to continue",
                      style: TextStyle(color: Colors.grey),
                    ),

                    const SizedBox(height: 30),

                    TextFormField(
                      controller: _nameController,
                      decoration: InputDecoration(
                        prefixIcon: const Icon(Icons.person_outline),
                        labelText: 'Full Name',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Please enter your name';
                        }
                        return null;
                      },
                    ),

                    const SizedBox(height: 18),

                    TextFormField(
                      controller: _emailController,
                      obscureText: false,
                      decoration: InputDecoration(
                        prefixIcon: const Icon(Icons.email_outlined),
                        labelText: 'Email',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      keyboardType: TextInputType.emailAddress,
                      onChanged: (val) {
                        if (val.contains('@')) {
                          setState(() => showUsernameField = true);
                        }
                      },
                      validator: (value) {
                        if (value == null || value.trim().isEmpty) {
                          return 'Please enter your email';
                        }
                        if (!RegExp(r'^[^@]+@[^@]+\.[^@]+').hasMatch(value)) {
                          return 'Please enter a valid email address';
                        }
                        return null;
                      },
                    ),

                    const SizedBox(height: 26),
                    if (showUsernameField)
                      TextFormField(
                        controller: _usernameController,
                        decoration: InputDecoration(labelText: 'Username (optional)'),
                      ),
                    if (showUsernameField) 
                    
                    const SizedBox(height: 16),

                    TextFormField(
                      controller: _passwordController,
                      obscureText: !_isPasswordVisible,
                      decoration: InputDecoration(
                        labelText: 'Password',
                        suffixIcon: IconButton(
                          icon: Icon(
                            _isPasswordVisible
                                ? Icons.visibility
                                : Icons.visibility_off,
                          ),
                          onPressed: () {
                            setState(() {
                              _isPasswordVisible = !_isPasswordVisible;
                            });
                          },
                        ),
                      ),
                      validator: (value) {
                        if (value == null || value.length < 6) {
                          return 'Password must be at least 6 characters long';
                        }
                        return null;
                      },
                    ),

                    SizedBox(height: 16),

                    TextFormField(
                      controller: _confirmController,
                      decoration: InputDecoration(labelText: 'Confirm Password'),
                      obscureText: true,
                      validator: (value) {
                        if (value != _passwordController.text) {
                          return 'Passwords do not match';
                        }
                        return null;
                      },
                    ),
                    SizedBox(height: 24),

                    SizedBox(
                      width: double.infinity,
                      height: 52,
                      child: ElevatedButton(
                        onPressed: _isLoading ? null : _submitForm,
                        style: ElevatedButton.styleFrom(
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                        child: const Text("Register"),
                      ),
                    ),

                    const SizedBox(height: 14),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
//   @override
//   Widget build(BuildContext context) {
//     return Center(
//       child: ConstrainedBox(
//         constraints: BoxConstraints(maxWidth: 400),
//         child: Form(
//           key: _formKey,
//           child: Column(
//             crossAxisAlignment: CrossAxisAlignment.stretch,
//             children: <Widget>[
//               Text(
//                 'Create Account',
//                 textAlign: TextAlign.center,
//                 style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
//               ),
//               const SizedBox(height: 24),

//               TextFormField(
//                 controller: _nameController,
//                 decoration: InputDecoration(labelText: 'Full Name'),
//                 validator: (value) {
//                   if (value == null || value.isEmpty) {
//                     return 'Please enter your name';
//                   }
//                   return null;
//                 },
//               ),
//               SizedBox(height: 16),

//               TextFormField(
//                 controller: _emailController,
//                 decoration: InputDecoration(labelText: 'Email'),
//                 keyboardType: TextInputType.emailAddress,
//                 onChanged: (val) {
//                   if (val.contains('@')) {
//                     setState(() => showUsernameField = true);
//                   }
//                 },
//                 validator: (value) {
//                   if (value == null || value.trim().isEmpty) {
//                     return 'Please enter your email';
//                   }
//                   if (!RegExp(r'^[^@]+@[^@]+\.[^@]+').hasMatch(value)) {
//                     return 'Please enter a valid email address';
//                   }
//                   return null;
//                 },
//               ),
//               SizedBox(height: 16),

//               if (showUsernameField)
//                 TextFormField(
//                   controller: _usernameController,
//                   decoration: InputDecoration(labelText: 'Username (optional)'),
//                 ),
//               if (showUsernameField) const SizedBox(height: 16),

//               TextFormField(
//                 controller: _passwordController,
//                 obscureText: !_isPasswordVisible,
//                 decoration: InputDecoration(
//                   labelText: 'Password',
//                   suffixIcon: IconButton(
//                     icon: Icon(
//                       _isPasswordVisible
//                           ? Icons.visibility
//                           : Icons.visibility_off,
//                     ),
//                     onPressed: () {
//                       setState(() {
//                         _isPasswordVisible = !_isPasswordVisible;
//                       });
//                     },
//                   ),
//                 ),
//                 validator: (value) {
//                   if (value == null || value.length < 6) {
//                     return 'Password must be at least 6 characters long';
//                   }
//                   return null;
//                 },
//               ),
//               SizedBox(height: 16),

//               TextFormField(
//                 controller: _confirmController,
//                 decoration: InputDecoration(labelText: 'Confirm Password'),
//                 obscureText: true,
//                 validator: (value) {
//                   if (value != _passwordController.text) {
//                     return 'Passwords do not match';
//                   }
//                   return null;
//                 },
//               ),
//               SizedBox(height: 24),

//               ElevatedButton(
//                 onPressed: _isLoading ? null : _submitForm,
//                 style: ElevatedButton.styleFrom(
//                   padding: const EdgeInsets.symmetric(vertical: 14),
//                   textStyle: const TextStyle(fontSize: 18),
//                 ),
//                 child: _isLoading
//                     ? const CircularProgressIndicator(color: Colors.white)
//                     : const Text('Register'),
//               ),
//             ],
//           )
//         ),
//       )
//     );
//   }
// }