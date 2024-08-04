/**
* This code was generated by v0 by Vercel.
* @see https://v0.dev/t/F2tmwMX9pKA
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
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

export function About() {
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
    <main>
      <section className="py-20 bg-white dark:bg-zinc-800" id="about">
        <div
          className="container mx-auto px-6 grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <h1 className="text-4xl font-bold mb-4 text-zinc-800">About GenAI</h1>
            <p className="text-lg text-zinc-800 dark:text-zinc-400 mb-8">
              GenAI is a cutting-edge platform that empowers developers and researchers to harness the power of
              generative artificial intelligence. Our mission is to democratize access to state-of-the-art AI models
              and tools, enabling users to create innovative solutions across various domains.
            </p>
            <p className="text-lg text-zinc-800 dark:text-zinc-400 mb-8">
              With our extensive collection of high-quality datasets and powerful models, we provide a comprehensive
              ecosystem for training, fine-tuning, and deploying generative AI applications. Whether you're working on
              natural language processing, computer vision, speech recognition, or any other AI-driven task, GenAI has
              the resources you need to succeed.
            </p>
          </div>
          <div>
            <img
              alt="About Image"
              className="rounded-lg"
              height={400}
              src="/placeholder.svg"
              style={{
                aspectRatio: "600/400",
                objectFit: "cover",
              }}
              width={600} />
          </div>
        </div>
      </section>
      <section className="py-20" id="features">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-4 text-black dark:text-white">Features</h2>
          <p className="text-lg text-zinc-800 dark:text-gray-400 mb-8">
            Unlock the full potential of generative AI with our comprehensive suite of features, designed to
            streamline your workflow and empower you to create groundbreaking applications with ease.
          </p>
          <div className="grid gap-8">
            <div>
              <h3 className="text-xl font-bold mb-2">Training</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Train your models with ease using our powerful and intuitive training platform. Leverage
                state-of-the-art techniques, advanced optimization algorithms, and scalable infrastructure to achieve
                optimal performance and accuracy.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Learn More</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Evaluation</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Evaluate your models' performance with our comprehensive suite of evaluation tools. Gain insights into
                your models' strengths and weaknesses, and make informed decisions to improve their performance.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Learn More</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Fine-Tuning</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Fine-tune your models to achieve optimal performance for your specific use case. Our fine-tuning tools
                allow you to leverage pre-trained models and adapt them to your unique requirements, saving time and
                resources.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Learn More</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Build & Deploy</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Build and deploy your generative AI applications with ease. Our platform offers seamless integration
                with popular frameworks and tools, enabling you to quickly and efficiently bring your ideas to life.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Learn More</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Model Marketplace</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Explore our extensive collection of pre-trained models and find the perfect solution for your project.
                Our Model Marketplace offers a wide range of models for various tasks and domains, ensuring you have
                access to the best tools for your needs.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Explore Marketplace</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Data Marketplace</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Access our vast collection of high-quality datasets to power your generative AI models. Our Data
                Marketplace offers a diverse range of datasets spanning various modalities and domains, ensuring you
                have access to the data you need to train accurate and robust models.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Explore Marketplace</Button>
              </div>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-2">Collaboration</h3>
              <p className="text-zinc-800 dark:text-gray-400 mb-4">
                Collaborate seamlessly with your team and leverage the power of collective intelligence. Our platform
                offers a suite of collaboration tools, enabling you to share models, datasets, and insights, fostering
                a collaborative environment for innovation.
              </p>
              <div className="flex justify-start mt-4">
                <Button variant="outline">Learn More</Button>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section className="py-20 bg-zinc-800 dark:bg-gray-800" id="contact">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-4 text-white dark:text-white">Contact Us</h2>
          <p className="text-lg text-zinc-400 dark:text-gray-400 mb-8">
            We'd love to hear from you! Whether you have a question, need assistance, or want to explore partnership
            opportunities, our team is here to help.
          </p>
          <div className="flex flex-col items-center">
            <form className="w-full max-w-md space-y-4">
              <div>
                <Label className="text-white" htmlFor="name">
                  Name
                </Label>
                <Input
                  className="border-blue-600 text-white placeholder:text-zinc-400"
                  id="name"
                  placeholder="Enter your name"
                  type="text" />
              </div>
              <div>
                <Label className="text-white" htmlFor="email">
                  Email
                </Label>
                <Input
                  className="border-blue-600 text-white placeholder:text-zinc-400"
                  id="email"
                  placeholder="Enter your email"
                  type="email" />
              </div>
              <div>
                <Label className="text-white" htmlFor="message">
                  Message
                </Label>
                <Textarea
                  className="border-blue-600 text-white placeholder:text-zinc-400"
                  id="message"
                  placeholder="Enter your message" />
              </div>
              <Button className="bg-blue-600 hover:bg-sky-400" type="submit">
                Submit
              </Button>
            </form>
          </div>
        </div>
      </section>
      <section className="py-20 bg-white dark:bg-zinc-900" id="registration">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h2 className="text-3xl font-bold mb-4 text-zinc-800 dark:text-zinc-200">Register Today</h2>
              <p className="text-lg text-zinc-800 dark:text-zinc-400 mb-8">
                Join our community of AI enthusiasts and gain access to exclusive datasets, models, and features. Get
                started today and unlock the full potential of GenAI.
              </p>
              <form className="w-full max-w-md space-y-4">
                <div>
                  <Label className="text-zinc-800 dark:text-zinc-200" htmlFor="name">
                    Name
                  </Label>
                  <Input id="name" placeholder="Enter your name" type="text" />
                </div>
                <div>
                  <Label className="text-zinc-800 dark:text-zinc-200" htmlFor="email">
                    Email
                  </Label>
                  <Input id="email" placeholder="Enter your email" type="email" />
                </div>
                <div>
                  <Label className="text-zinc-800 dark:text-zinc-200" htmlFor="password">
                    Password
                  </Label>
                  <Input id="password" placeholder="Enter your password" type="password" />
                </div>
                <Button className="bg-emerald-800 hover:bg-green-400" type="submit">
                  Register
                </Button>
              </form>
            </div>
            <div>
              <h3 className="text-2xl font-bold mb-4 text-white dark:text-gray-200">Why Join GenAI?</h3>
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <p className="text-zinc-800 dark:text-zinc-400">Access to a vast collection of datasets</p>
                </div>
                <div className="flex items-center space-x-4">
                  <p className="text-zinc-800 dark:text-zinc-400">State-of-the-art AI models for various tasks</p>
                </div>
                <div className="flex items-center space-x-4">
                  <p className="text-zinc-800 dark:text-zinc-400">Cutting-edge features for AI development</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
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
