/**
* This code was generated by v0 by Vercel.
* @see https://v0.dev/t/BTRviuIyEkJ
* Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
*/

/** Add fonts into your Next.js project:

import { Archivo } from 'next/font/google'
import { Comfortaa } from 'next/font/google'

archivo({
  subsets: ['latin'],
  display: 'swap',
})

comfortaa({
  subsets: ['latin'],
  display: 'swap',
})

To read more about using these font, please visit the Next.js documentation:
- App Directory: https://nextjs.org/docs/app/building-your-application/optimizing/fonts
- Pages Directory: https://nextjs.org/docs/pages/building-your-application/optimizing/fonts
**/
'use client';
import React, {useState} from 'react';
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import api from './api'

  

export function Register() {
  const host = 'http://localhost:8000/' 
  const [registrationData, setRegistrationData] = useState({
    uname: '',
    email: '',
    pwd: '',
    pwd_conf: ''
  });
  



  const handleInputChange = (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setRegistrationData({
      ...registrationData,
      [event.target.name]: value,
    });
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    try {
    await api.post('/register/', registrationData);
    setRegistrationData({
      uname: '',
      email: '',
      pwd: '',
      pwd_conf: ''
    });
    }
    catch (error) {
      console.log('Registration failed', error);
    }
  };
  return (<>
   <header
      className="flex items-center justify-between px-6 py-4 bg-blue-950 text-white">
      <div className="flex items-center">
        <Link className="flex items-center" href="/">
          <BrainIcon className="h-8 w-8 mr-2" />
          <span className="text-xl font-bold">GenAI</span>
        </Link>
      </div>
      <nav className="flex items-center space-x-6">
        <Link className="hover:underline" href="#">
          Datasets
        </Link>
        <Link className="hover:underline" href="#">
          Models
        </Link>
        <Link className="hover:underline" href="/about">
          About
        </Link>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button className="hover:bg-blue-800 text-white border" variant="ghost">
              Account
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem>
              <Link href="/register">Register</Link>
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Link href="/login">Login</Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              <Link href="#">Account Settings</Link>
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Link href="#">Data</Link>
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Link href="#">Models</Link>
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Link href="#">Logout</Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </nav>
    </header>
    <main className="bg-zinc-900 min-h-screen flex items-center justify-center">
      <div className="bg-zinc-200 rounded-lg shadow-lg p-8 max-w-md w-full">
        <h2 className="text-3xl font-bold mb-6 text-center">Register</h2>
        <form onSubmit={handleFormSubmit}>
          <div className="mb-4">
            <Label className="block mb-2 text-zinc-800" htmlFor="uname">
              Name
            </Label>
            <Input id="uname" placeholder="Enter your name" type="text" name="uname" value={registrationData.uname} onChange={handleInputChange}/>
          </div>
          <div className="mb-4">
            <Label className="block mb-2 text-zinc-800" htmlFor="email">
              Email
            </Label>
            <Input id="email" placeholder="Enter your email" type="text" name="email" value={registrationData.email} onChange={handleInputChange}/>
          </div>
          <div className="mb-4">
            <Label className="block mb-2 text-zinc-800" htmlFor="password">
              Password
            </Label>
            <Input id="pwd" placeholder="Enter your password" type="text" name="pwd" value={registrationData.pwd} onChange={handleInputChange}/>
          </div>
          <div className="mb-4">
            <Label className="block mb-2 text-zinc-800" htmlFor="pwd_conf">
              Confirm Password
            </Label>
            <Input id="pwd_conf" placeholder="Confirm your password" type="text" name="pwd_conf" value={registrationData.pwd_conf} onChange={handleInputChange}/>
          </div>
          <Button className="w-full bg-zinc-600 text-white hover:bg-zinc-500" type="submit">
            Register
          </Button>
        </form>
        <div className="mt-4 text-center text-zinc-800">
          Already have an account?
          <Link className="text-blue-600 hover:underline" href="/login">
            Login
          </Link>
        </div>
      </div>
    </main>
    <footer className="bg-blue-950 text-white py-6">
      <div className="container mx-auto px-6 grid grid-cols-4 gap-6">
        <div>
          <h3 className="text-lg font-bold mb-4">Company</h3>
          <ul>
            <li>
              <Link className="hover:underline" href="#">
                About
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Contact
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Blog
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <h3 className="text-lg font-bold mb-4">Datasets</h3>
          <ul>
            <li>
              <Link className="hover:underline" href="#">
                Text
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Images
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Audio
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Structured
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Unstructured
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Custom
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <h3 className="text-lg font-bold mb-4">Models</h3>
          <ul>
            <li>
              <Link className="hover:underline" href="#">
                Llama-3
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Stable Diffusion
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                ViT
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                DeTr
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Whisper
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                *XTTS
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <h3 className="text-lg font-bold mb-4">Product</h3>
          <ul>
            <li>
              <Link className="hover:underline" href="#">
                Training
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Evaluation
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Fine-Tuning
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Build & Deploy
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Model Marketplace
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Data Marketplace
              </Link>
            </li>
            <li>
              <Link className="hover:underline" href="#">
                Collaboration
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </footer>
  </>);
}

function BrainIcon(props) {
  return (
    (<svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round">
      <path
        d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
      <path
        d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
      <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
      <path d="M6 18a4 4 0 0 1-1.967-.516" />
      <path d="M19.967 17.484A4 4 0 0 1 18 18" />
    </svg>)
  );
}
