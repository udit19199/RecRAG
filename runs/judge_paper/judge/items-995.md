Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 1 items.

=== ITEM id=1ef42349fa97f9ba ===
Gold sentences and facts:
Sentence 0: Because GPT-4 is more likely to follow the instructions of “jailbreaking” prompts, the co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
  Fact 0.0: GPT-4 is more likely to follow the instructions of 'jailbreaking' prompts.
  Fact 0.1: The co-authors claim that GPT-4 can be more easily prompted than other LLMs to spout toxic, biased text.
Sentence 1: Its earlier actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.
  Fact 1.0: The entity's earlier actions were focused on concerns about the spread of illegal content.
  Fact 1.1: The entity's earlier actions were focused on concerns about the spread of disinformation related to the Israel-Hamas war.

Retrieved items:
<item 1>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT can also show bias in its responses, answering in sexist and racist, overtly Anglocentric ways — or regurgitating portions of the data that it was
trained on.

Lawyers have been sanctioned after using ChatGPT to assist in writing motions, discovering — too late — that ChatGPT invented fake lawsuit citations. And scores
of authors have sued OpenAI over the chatbot regurgitating portions of their work — and not receiving compensation for it.

So what comes next? What might ChatGPT’s second year hold, if not more of the same?

Interestingly — and fortunately — some of the more dire predictions about ChatGPT didn’t come to pass. Some researchers feared the chatbot would be used to
generate disinformation on a massive scale, while others sounded the alarm over ChatGPT’s phishing email-, spam- and malware-generating potential.
</item 1>

<item 2>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
users at launch.

When using the mobile version of ChatGPT, the app will sync your history across devices — meaning it will know what you’ve previously searched for via its web
interface, and make that accessible to you. The app is also integrated with Whisper, OpenAI’s open source speech recognition system, to allow for voice input.

Hackers are using ChatGPT lures to spread malware on Facebook

Meta said in a report on May 3 that malware posing as ChatGPT was on the rise across its platforms. The company said that since March 2023, its security teams
have uncovered 10 malware families using ChatGPT (and similar themes) to deliver malicious software to users’ devices.

“In one case, we’ve seen threat actors create malicious browser extensions available in official web stores that claim to offer ChatGPT-based tools,” said Meta
security engineers Duc H. Nguyen and Ryan Victory in a blog post. “They would then promote these malicious extensions on social media and through sponsored
search results to trick people into downloading malware.”
</item 2>

<item 3>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q2: Electric Boog-AI-loo

Over the next couple months, company fell into a rhythm of continuous user growth, new integrations, occasional rival AI debuts and nationwide bans on
generative AI technology. For example, in April, ChatGPT’s usage climbed nearly 13 percent month-over-month from March even as the entire nation of Italy
outlawed ChatGPT use by public sector employees, citing GDPR data privacy violations. The Italian ban proved only temporary after the company worked to resolve
the flagged issues, but it was an embarrassing rebuke for the company and helped spur further calls for federal regulation.

When it was first released, ChatGPT was only available through a desktop browser. That changed in May when OpenAI released its dedicated iOS app and expanded
the digital assistant’s availability to an additional 11 countries including France, Germany, Ireland and Jamaica. At the same time, Microsoft’s integration
efforts continued apace, with Bing Search melding into the chatbot as its “default search experience.”
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
They’re all chasing after a customer service software market that’ll be worth $58.1 billion by 2023, assuming the rather optimistic prediction from Acumen
Research comes true.

Here are the YC-backed startups that caught our eye:

Yuma, whose customer demographic is primarily Shopify merchants, provides ChatGPT-like AI systems that integrate with help desk software, suggesting drafts of
replies to customer tickets.

Baselit, which uses one of OpenAI’s text-understanding models to allow businesses to embed chatbot-style analytics for their customers.

Lasso customers send descriptions or videos of the processes they’d like to automate and the company combines ChatGPT-like interface with robotic process
automation (RPA) and a Chrome extension to build out those automations.

BerriAI, whose platform is designed to help developers spin up ChatGPT apps for their organization data through various data connectors.

Italy orders ChatGPT to be blocked

OpenAI has started geoblocking access to its generative AI chatbot, ChatGPT, in Italy.
</item 4>

<item 5>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
“Foundation models, of course, are part of the general purpose AI models. These are models that can be used for a very large variety of tasks, they can also be
integrated in systems. To give you a concrete example, the general purpose AI model would be GPT-4 and the general purpose AI system would be ChatGPT — where
GPT-4 is integrated in ChatGPT.”

As we reported earlier, the deal agreed by the bloc’s co-legislators includes a low risk tier and a high risk tier for regulating so-called general purpose AIs
(GPAIs) — such as models behind the viral boom in generative AI tools like OpenAI’s ChatGPT. The trigger for high risk rules to apply on generative AI
technologies is determined by an initial threshold set out in the law.
</item 5>

<item 6>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
However, what makes the app different from the default experience or the dozens of generic AI chat apps now available are the characters offered which you can
use to engage with SuperChat’s AI features.

Italy gives OpenAI to-do list for lifting ChatGPT suspension order

Italy’s data protection watchdog has laid out what OpenAI needs to do for it to lift an order against ChatGPT issued at the end of last month — when it said it
suspected the AI chatbot service was in breach of the EU’s GSPR and ordered the U.S.-based company to stop processing locals’ data.

The DPA has given OpenAI a deadline — of April 30 — to get the regulator’s compliance demands done. (The local radio, TV and internet awareness campaign has a
slightly more generous timeline of May 15 to be actioned.)
</item 6>

<item 7>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
He’s also ripped up a bunch of legacy content moderation policies and slashed in-house enforcement teams while promoting a decentralized, crowdsourced
alternative (rebranded as Community Notes), which essentially outsources responsibility for dealing with tricky issues like disinformation to users in what
looks suspiciously like another gambit to eke out extra engagement and farm confusion by applying a philosophy of extreme relativism so culture warriors are
encouraged to keep forever fighting for their own “truth” in the comments.

Oh and he also pulled X out of the EU’s Code of Practice on Disinformation earlier this year in a very clear thumb of the nose to EU regulators.

Urgent letter to Musk

In an “urgent” letter to Musk today, which the EU’s internal market commissioner Thierry Breton also shared on X, the bloc has sent the strongest signal yet it
believes Musk’s platform is in breach of the DSA — although this is by no means Musk’s first warning.
</item 7>

<item 8>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
Whether this is capturing GPT-4 or Gemini or others we are not here now to assert — because also, in our framework, it is the companies that would have to come
and self assess what the amount of FLOPs or the computing capacity they have used. But, of course, if you read the scientific literature, many will point to
these numbers as being very much the most advanced models at the moment. We will see what the companies will assess because they’re the best placed to make this
assessment.”

“The rules have not been written keeping in mind certain companies,” they added. “They’ve really been written with the idea of defining the threshold — which,
by the way, may change because we have the possibility to be empowered to change this threshold on the basis of technological evolution. It could go up, it
could go down and we could also develop other benchmarks that in the future will be the more appropriate to benchmark the different moments.”
</item 8>

<item 9>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“In this context, 832 videos related to the war have been fact-checked, of which 211 have been removed,” Jourova noted.

The EU also flagged reporting by Microsoft that told it Bing Search had either promoted information or downgraded questionable information in relation to almost
800,000 search queries related to the Ukraine crisis.

Jourova’s speech also highlighted a couple of other areas where she urged Code signatories to go further — calling (yet again) for more consistent moderation
and investment in fact-checking, especially in smaller Member States and languages.

She also criticized platforms over access to data, saying they must step up efforts to make sure researchers are empowered to scrutinize disinformation flows
“and contribute to the necessary transparency”.

Both are areas where X/Twitter under new owner, Elon Musk, has moved out of step with EU expectations on countering disinformation.

Twitter (now X) was an original signatory to the disinformation Code but Musk took the platform out of the initiative back in May, as critical scrutiny of his
actions dialled up in the EU.
</item 9>

<item 10>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
EU regulators are also set to assess Musk & co’s claims that replacing a formal content moderation function with crowdsourced opinions is a valid approach to
tackling diverse information risks; and, consequently, to look into how much Musk’s gutting of content moderation teams and processes has cost X’s users in
terms of harm exposure.

To say the Commission’s investigation looks fascinating and timely is an understatement.

EU lawmakers can certainly bask in the glow of being prepared for the possibility of an erratic billionaire running a wrecking ball through a major social
platform. But of course they will need to follow through — and it’s clear the real work is just starting. So this probe looks set to be a test of how much
stomach the EU has to actually enforce the rules it loves to lay down on paper.

The investigation may also test Musk’s mettle for what could be an expensive head-on clash with EU regulators.
</item 10>

<item 11>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s at the root of the research we shared from the institute last week. Noted senior research scientist Benjamin Burchfiel:

We’ve seen some big progress with the advent of [large language models], using them to impart this high level of cognitive intelligence into robots. If you have
a robot that picks up a thing, now instead of having to specify an object, you can tell it to pick up the can of Coke. Or you can tell it to pick up the shiny
object, or you can do the same thing and do it in French. That’s really great, but if you want a robot to plug in a USB device or pick up a tissue, those models
just don’t work. They’re really useful, but they don’t solve that part of the problem. We’re focused on filling in that missing piece, and the thing we’re
really excited about now is that we actually have a system and that the fundamentals are correct.
</item 11>

<item 12>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
A rule of law clash between Musk and the EU has looked increasingly inevitable in recent months — and, indeed, has been predicted by some industry watchers
since rumors of the erratic billionaire’s plan to take over Twitter emerged last year.

Reminder: Penalties for confirmed breaches of the DSA can be as high as 6% of global annual turnover. The bloc also has powers, in extremis, which could result
in access to X being shut off in the region if the platform repeatedly fails to correct course. So if Musk fails to satisfy EU regulators, he could face serious
consequences for what remains a highly indebted business.

In his letter to Musk, Breton writes that the EU has “indications” that X is being used to disseminate illegal content and disinformation in the EU” following
Saturday’s attacks. He then goes on to remind the company of the DSA’s “very precise obligations” vis-à-vis content moderation.
</item 12>

<item 13>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
The test involved asking ChatGPT to repeat the word “poem” forever, among other words, which over time led the chatbot to churn out private information like
email addresses and phone numbers.

ChatGPT and other AI chatbots are fueling an increase in phishing emails

According to a new report by SlashNext, there’s been a 1,265% increase in malicious phishing emails since Q4 of 2022. The report alleges that AI tools like
ChatGPT are being prominently used by cybercriminals to write compelling and sophisticated phishing emails.

South Africa officials investigate if President Cyril Ramaphosa used ChatGPT to write a speech

Following speculation, social media users fed portions of Ramaphosa’s November 21 speech in Johannesburg through AI detectors, alleging parts of it may have
been written with ChatGPT. South African presidency spokesperson Vincent Magwenya refuted the claims, and local officials are investigating.
</item 13>

<item 14>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
That doesn’t sound a lot like the “mission” the board wants to advance. Then again, this shift didn’t happen today, and it certainly doesn’t seem like a
plausible reason for abruptly firing the CEO and a few others on a beautiful fall Friday afternoon.

6. What about IP and legal liability?

Altman told me at OpenAI’s Dev Day earlier this month that the company doesn’t want to incur any copyright problems by using (as I had asked about) datasets of
pirated books. But a lot of research I’ve been reading contradicts that, as does pretty much every AI data scientist I talk to. It’s exceedingly hard to imagine
that OpenAI built GPT-3 with the copyrighted books database (as seems to be the case) but not GPT-4 or succeeding models. (I was going to write this up next
week, so thanks OpenAI board for eating my lunch.)
</item 14>

<item 15>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
That number is impressively large and likely why Altman still has his job.

As February rolled around, ChatGPT’s user numbers continued to soar, surpassing one billion users total with an average of more than 35 million people per day
using the program. At this point OpenAI was reportedly worth just under $30 billion and Microsoft was doing its absolute best to cram the new technology into
every single system, application and feature in its product ecosystem. ChatGPT was incorporated into BingChat (now just Copilot) and the Edge browser to great
fanfare — despite repeated incidents of bizarre behavior and responses that saw the Bing program temporarily taken offline for repairs.

Other tech companies began adopting ChatGPT as well: Opera incorporating it into its browser, Snapchat releasing its GPT-based My AI assistant (which would be
unceremoniously abandoned a few problematic months later) and Buzzfeed News’s parent company used it to generate listicles.
</item 15>

<item 16>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 16>

<item 17>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
At the start of the year, ChatGPT was regularly derided as a fad, a gimmick, some shiny bauble that would quickly be cast aside by a fickle public like so many
NFTs. Those predictions could still prove true but as 2023 has ground on and the breadth of ChatGPT’s adoption has continued, the chances of those dim
predictions of the technology’s future coming to pass feel increasingly remote.

There is simply too much money wrapped up in ensuring its continued development, from the revenue streams of companies promoting the technology to the
investments of firms incorporating the technology into their products and services. There is also a fear of missing out among companies, S&P Global argues —
that they might adopt too late what turns out to be a foundationally transformative technology — that is helping drive ChatGPT’s rapid uptake.

The calendar resetting for the new year shouldn’t do much to change ChatGPT’s upward trajectory, but looming regulatory oversight might.
</item 17>

<item 18>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT got an overall three-star rating in the report, with its lowest ratings relating to transparency, privacy, trust and safety.

OpenAI blames DDoS attack for ChatGPT outage

OpenAI confirmed that a DDoS attack was behind outages affecting ChatGPT and its developer tools. ChatGPT experienced sporadic outages for about 24 hours,
resulting in users being unable to log into or use the service.

OpenAI debuts GPT-4 Turbo

OpenAI unveiled GPT-4 Turbo at its first-ever OpenAI DevDay conference. GPT-4 Turbo comes in two versions: one that’s strictly text-analyzing and another that
understands the context of both text and images.

GPT-4 gets a fine-tuning

As opposed to the fine-tuning program for GPT-3.5, the GPT-4 program will involve more oversight and guidance from OpenAI teams, the company says — largely due
to technical hurdles.
</item 18>

<item 19>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Reached for a response to the EU probe, X’s Joe Benarroch — who was brought into the company by CEO Linda Yaccarino relatively recently to work on business
operations — said:

X remains committed to complying with the Digital Services Act, and is cooperating with the regulatory process. It is important that this process remains free
of political influence and follows the law. X is focused on creating a safe and inclusive environment for all users on our platform, while protecting freedom of
expression, and we will continue to work tirelessly towards this goal.

While X under Musk has very evidently been flying the ship in the opposite direction to the responsible governance the DSA intends to encourage — hence
attracting so many warnings from EU lawmakers, even long before today’s formal probe — it’s notable X recently started offering a research program for outsiders
to get data to systemic risks arising on the platform in the EU. However the Commission obviously has doubts X has gone far enough on the transparency front to
meet the DSA’s bar.
</item 19>

<item 20>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
While there is a more…nefarious side to ChatGPT, it’s clear that AI tools are not going away anytime soon. Since its initial launch nearly a year ago, ChatGPT
has hit 100 million weekly active users, and OpenAI is heavily investing in it.

Prior to the leadership chaos, on November 6, OpenAI held its first developer conference: OpenAI DevDay. During the conference, it announced a slew of updates
coming to GPT, including GPT-4 Turbo (super-charged versions of GPT-4, its latest language-writing model), a multimodal API and a GPT store where users can
create and monetize their own custom versions of GPT.

GPT-4, which can write more naturally and fluently than previous models, remains largely exclusive to paying ChatGPT users. But you can access GPT-4 for free
through Microsoft’s Bing Chat in Microsoft Edge, Google Chrome and Safari web browsers.
</item 20>

<item 21>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
Asking Musk to be “effective” against disinformation is a bit like asking the sea to stop moving. But, well, this is how the regulatory dance must go (and after
the dance comes the denouement — which, if a DSA breach is confirmed, means enforcement. And actual penalties might be a bit harder for Musk to troll).

In the meantime, the EU has asked Musk to make contact with relevant law enforcement authorities and Europol — and “ensure that you respond promptly to their
requests.” Breton also flags some unspecified additional DSA compliance issues he says his team will be contacting Musk’s about “shortly” — “with a specific
request.”. (We’ve asked the EU what else it’s concerned about on X and will update this report with any response.)

The bloc has given Musk a deadline of 24 hours to respond to its asks at this point — stipulating his answer will be added to its assessment file on X’s
compliance with the DSA.
</item 21>

<item 22>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
For example, having the model always respond in a given language. Or improving the model’s ability to consistently format responses, as well as hone the “feel”
of the model’s output, like its tone, so that it better fits a brand or voice. Most notably, fine-tuning enables OpenAI customers to shorten text prompts to
speed up API calls and cut costs.

OpenAI is partnering with Scale AI to allow companies to fine-tune GPT-3.5. However, it is unclear whether OpenAI is developing an in-house tuning tool that is
meant to complement platforms like Scale AI or serve a different purpose altogether.
</item 22>

<item 23>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
X should have “proportionate and effective mitigation measures” in place to tackle “the risks to public security and civic discourse stemming from
disinformation,” he also says.

Instead, the platform appears to be turning into an engine of disinformation — one that’s demonstrably enabled the very swift amplification of a smorgasbord of
toxic fakes around the Israel-Hamas war. Falsehoods that may be trying to manipulate opinion around the conflict, or otherwise exploit horrific events to drive
engagement (clickbait) or for even more bleak, cynical and potentially harmful ends.

“Public media and civil society organisations widely report instances of fake and manipulated images and facts circulating on your platform in the EU, such as
repurposed old images of unrelated armed conflicts or military footage that actually originated from video games. This appears to be manifestly false or
misleading information,” writes Breton. “I therefore invite you to urgently ensure that your systems are effective, and report on the crisis measures taken to
my team.”
</item 23>

<item 24>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
It also puts legal obligation on larger platforms — including X — to mitigate risks from disinformation. So the fast-moving and bloody events unfolding in
Israel and Gaza are offering a real-world test of whether the EU’s rebooted rulebook is big and beefy enough to tackle X’s most notorious shitposter. Who, since
last fall, is also the platform’s owner.

Since taking over Twitter (as it was then), Musk has painted the largest target on X when it comes to DSA enforcement on account of a series of changes he’s
pushed out that make it harder for users to locate quality information on X.

This includes ending legacy account verification and turning the Blue Check system into a game of pay-to-play.
</item 24>

<item 25>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU said the probe will also look at the effectiveness of “related policies mitigating risks to civic discourse and electoral processes”

— with a focus on the effectiveness of its “Community Notes” system of crowdsourcing fact-checking to aggregated views of users, who may vote to label disputed
tweets with additional context. The EU said the probe will also look at the Transparency requirements — with a focus on “suspected shortcomings” by X in giving
researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected shortcomings in its ads repository (aka the ads transparency
library the regulation also mandates)

— with a focus on “suspected shortcomings” by X in giving researchers access to publicly accessible data as mandated by Article 40 of the DSA, and suspected
shortcomings in its ads repository (aka the ads transparency library the regulation also mandates) Suspected deceptive design elements in X’s user interface —
“notably in relation to checkmarks linked to certain subscription products, the so-called Blue checks”

“If proven,
</item 25>

<item 26>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT can engage with a range of topics, including programming, TV scripts and scientific concepts. Writers everywhere rolled their eyes at the new
technology, much like artists did with OpenAI’s DALL-E model, but the latest chat-style iteration seemingly broadened its appeal and audience.

FAQs:

What is ChatGPT? How does it work?

ChatGPT is a general-purpose chatbot that uses artificial intelligence to generate text after a user enters a prompt, developed by tech startup OpenAI. The
chatbot uses GPT-4, a large language model that uses deep learning to produce human-like text.

When did ChatGPT get released?

November 30, 2022 is when ChatGPT was released for public use.

What is the latest version of ChatGPT?

Both the free version of ChatGPT and the paid ChatGPT Plus are regularly updated with new GPT models. The most recent model is GPT-4.

Can I use ChatGPT for free?
</item 26>

<item 27>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 27>

<item 28>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Who owns the copyright on ChatGPT-created content or media?

The user who requested the input from ChatGPT is the copyright owner.

What lawsuits are there surrounding ChatGPT?

None specifically targeting ChatGPT. But OpenAI is involved in at least one lawsuit that has implications for AI systems trained on publicly available data,
which would touch on ChatGPT.

Are there issues regarding plagiarism with ChatGPT?

Yes. Text-generating AI models like ChatGPT have a tendency to regurgitate content from their training data.
</item 28>

<item 29>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Italy’s data protection authority has just put out a timely reminder that some countries do have laws that already apply to cutting edge AI: it has ordered
OpenAI to stop processing people’s data locally with immediate effect. The Italian DPA said it’s concerned that the ChatGPT maker is breaching the European
Union’s General Data Protection Regulation (GDPR), and is opening an investigation.

March 2023

1,100+ signatories signed an open letter asking all ‘AI labs to immediately pause for 6 months’

The letter’s signatories include Elon Musk, Steve Wozniak and Tristan Harris of the Center for Humane Technology, among others. The letter calls on “all AI labs
to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4.”

The letter reads:

Contemporary AI systems are now becoming human-competitive at general tasks,[3] and we must ask ourselves: Should we let machines flood our information channels
with propaganda and untruth?
</item 29>

<item 30>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT is AI-powered and utilizes LLM technology to generate text after a prompt.

Can ChatGPT write essays?

Yes.

Can ChatGPT commit libel?

Due to the nature of how these models work, they don’t know or care whether something is true, only that it looks true. That’s a problem when you’re using it to
do your homework, sure, but when it accuses you of a crime you didn’t commit, that may well at this point be libel.

We will see how handling troubling statements produced by ChatGPT will play out over the next few months as tech and legal experts attempt to tackle the fastest
moving target in the industry.

Does ChatGPT have an app?

Yes, there is now a free ChatGPT app that is currently limited to U.S. iOS users at launch. OpenAi says an android version is “coming soon.”

What is the ChatGPT character limit?

It’s not documented anywhere that ChatGPT has a character limit.
</item 30>

<item 31>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Researchers discover a way to make ChatGPT consistently toxic

A study co-authored by scientists at the Allen Institute for AI shows that assigning ChatGPT a “persona” — for example, “a bad person,” “a horrible person” or
“a nasty person” — through the ChatGPT API increases its toxicity sixfold. Even more concerning, the co-authors found having the conversational AI chatbot pose
as certain historical figures, gendered people and members of political parties also increased its toxicity — with journalists, men and Republicans in
particular causing the machine learning model to say more offensive things than it normally would.

The research was conducted using the latest version, but not the model currently in preview based on OpenAI’s GPT-4.

Y Combinator-backed startups are trying to build ‘ChatGPT for X’

YC Demo Day’s Winter 2023 batch features no fewer than four startups that claim to be building “ChatGPT for X.”
</item 31>

<item 32>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
A Pew Research survey from August showed that only 18% of Americans have ever tried ChatGPT, and that most who’ve tried it use the chatbot for entertainment
purposes or answering one-off questions. Teens might not be using ChatGPT all that often, either (despite what some alarmist headlines imply), with one poll
finding that only two in five teenagers have used the tech in the last six months.

ChatGPT’s limitations might be to blame.

While undeniably capable, ChatGPT is far from perfect, owing to the way it was developed and “taught.” Trained to predict the likeliest next word — or likeliest
next parts of words — by observing billions of examples of text from around the web, ChatGPT sometimes “hallucinates,” or writes answers that sound plausible
but aren’t factually correct. (ChatGPT’s hallucinating tendencies got its answers banned from the Q&A site Stack Overflow and from at least one academic
conference — and accused of defamation.)
</item 32>

<item 33>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
A person can also specify their family size, so the text-generating AI can give responses about meals, grocery and vacation planning accordingly.

The FTC is reportedly investigating OpenAI

The FTC is reportedly in at least the exploratory phase of investigation over whether OpenAI’s flagship ChatGPT conversational AI made “false, misleading,
disparaging or harmful” statements about people.

TechCrunch Reporter Devin Coldewey reports:

This kind of investigation doesn’t just appear out of thin air — the FTC doesn’t look around and say “That looks suspicious.” Generally a lawsuit or formal
complaint is brought to their attention and the practices described by it imply that regulations are being ignored. For example, a person may sue a supplement
company because the pills made them sick, and the FTC will launch an investigation on the back of that because there’s evidence the company lied about the side
effects.

OpenAI announced the general availability of GPT-4

Starting July 6, all existing OpenAI developers “with a history of successful payments” can access GPT-4.
</item 33>

<item 34>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
EU turns up the heat on X over illegal content in wake of Israel-Hamas war

The European Union is cranking up the heat on Elon Musk-owned X. Late Thursday, the Commission sent the company a formal request for more information — after
issuing a public warning Wednesday about reports that illegal content and disinformation targeting the Israel-Hamas war is circulating on the platform, and
Twitter’s CEO sending a high-level (non specific) response to that warning.

The move could prefigure the opening of a formal investigation of X under the bloc’s content moderation rulebook, the Digital Services Act (DSA). If the
Commission moves ahead on that, it will be the first investigation opened under the DSA since a compliance deadline for so called “very large online platforms”
(aka VLOPs) kicked in this summer.

Non-compliance under DSA rules, meanwhile, could trigger fines of up to 6% of annual turnover, plus potential blocking of a service for repeated infringements.
X owner Elon Musk last projected revenues of about $3 billion for this year.
</item 34>

<item 35>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
The EU also said then it was looking at X’s compliance across a number of areas, including policies and practices regarding notices on illegal content,
complaint handling, risk assessment and measures to mitigate the risks identified — and it gave X until the end of October to respond.

“On the basis of the preliminary investigation conducted so far, including on the basis of an analysis of the risk assessment report submitted by X in
September, X’s Transparency report published on 3 November, and X’s replies to a formal request for information, which, among others, concerned the
dissemination of illegal content in the context of Hamas’ terrorist attacks against Israel, the Commission has decided to open formal infringement proceedings
against X under the Digital Services Act,” the EU said today.
</item 35>

<item 36>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 36>

<item 37>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“I remind you that following the opening of a potential investigation and a finding of non-compliance, penalties can be imposed,” Breton adds, concluding the
letter.

We’ve asked the Commission to confirm whether it has opened an investigation into X’s DSA compliance over concerns it’s raised in the letter. Perhaps it’s going
to wait a day to see what his response is before taking that next dance step.

Musk’s extremely iterative — or just plain erratic/arbitrary — management style is so far from the responsive and responsible qualities the EU’s rulebook
demands of digital leaders it’s hard to see how this clash can end well for either side.

We contacted X for a response to the EU’s warnings about its DSA compliance but at press time the company had not responded — beyond firing back its usual
automated reply, which reads: “Busy now, please check back later.”
</item 37>

<item 38>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
But at the time of writing, Musk had engaged in a bit of a dance of his own with the news of the EU’s warning by responding to a critical tweet posted to X by
journalist Glenn Greenwald, who attacked the EU’s new “censorship law,” as he dubbed the DSA — which he claimed would be used to “punish X.”

In reply to Greenwald, Musk avoided expressing the same trenchant criticism of the EU’s approach but invited yet more relativism — writing: “Let the public hear
exactly what this disinformation consists of and decide for themselves.”

He then went on to sew doubt that disinformation is something that can be independently arbitrated at all, implying fact-checking is just a convenient exercise
for targeting different opinions — so essentially echoing Greenwald’s position — by claiming that “many times” the “official fact-checker” has been found making
false statements, before adding a rhetorically empty offer that “Maybe this is the case here, maybe not” for good, plausible deniability measure.
</item 38>

<item 39>
OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)
The first two categories, being real and not fictional, have a relatively easy-to-understand rubric. Their teams rate each model on four risk categories:
cybersecurity, “persuasion” (e.g., disinfo), model autonomy (i.e., acting on its own), and CBRN (chemical, biological, radiological, and nuclear threats; e.g.,
the ability to create novel pathogens).

Various mitigations are assumed: For instance, a reasonable reticence to describe the process of making napalm or pipe bombs. After taking into account known
mitigations, if a model is still evaluated as having a “high” risk, it cannot be deployed, and if a model has any “critical” risks, it will not be developed
further.

These risk levels are actually documented in the framework, in case you were wondering if they are to be left to the discretion of some engineer or product
manager.

For example, in the cybersecurity section, which is the most practical of them, it is a “medium” risk to “increase the productivity of operators .
</item 39>

<item 40>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Should we automate away all the jobs, including the fulfilling ones? Should we develop nonhuman minds that might eventually outnumber, outsmart, obsolete and
replace us? Should we risk loss of control of our civilization? Such decisions must not be delegated to unelected tech leaders. Powerful AI systems should be
developed only once we are confident that their effects will be positive and their risks will be manageable.

OpenAI connects ChatGPT to the internet

OpenAI launched plugins for ChatGPT, extending the bot’s functionality by granting it access to third-party knowledge sources and databases, including the web.
Available in alpha to ChatGPT users and developers on the waitlist, OpenAI says that it’ll initially prioritize a small number of developers and subscribers to
its premium ChatGPT Plus plan before rolling out larger-scale and API access.

OpenAI launches GPT-4, available through ChatGPT Plus

GPT-4 is a powerful image- and text-understanding AI model from OpenAI.
</item 40>

<item 41>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Instead, users will automatically be switched to models based on the prompt.

ChatGPT officially gets web search

OpenAI has formally launched its internet-browsing feature to ChatGPT, some three weeks after re-introducing the feature in beta after several months in hiatus.
The AI chatbot that has historically been limited to data up to September, 2021.

OpenAI integrates DALL-E 3 into ChatGPT

The integration means users don’t have to think so carefully about their text-prompts when asking DALL-E to create an image. Users will also now be able to
receive images as part of their text-based queries without having to switch between apps.

Microsoft-affiliated research finds flaws in GPT-4

A Microsoft-affiliated scientific paper looked at the “trustworthiness” — and toxicity — of LLMs, including GPT-4.
</item 41>

<item 42>
WTF is going on at OpenAI? We have theories (TechCrunch, 2023-11-18T00:09:53+00:00)
But one can never be sure.

4. Could it be a major security or privacy incident?

The idea that the company has experienced a major, perhaps pervasive, security issue is bolstered by the fact that Microsoft reportedly suspended use of ChatGPT
internally a few days ago. OpenAI subsequently stopped allowing new signups. If there was a serious security problem in its biggest product and Altman
downplayed it, that would obviously create distrust with the board.

There is also the potential for misuse at scale with the enormous amount of personal data that travels through OpenAI’s APIs and services.

Working against this theory is the fact that CTO Mira Murati was just elevated to interim CEO in Altman’s place. It seems unlikely that anything
security-related would go through the CEO and not the CTO, or that the two would be at odds to the point where one could be fired like this and the other
swapped in to clean up the mess. As the board’s statement notes, Murati is in charge of product and safety, among other things.
</item 42>

<item 43>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
March saw more of the same, with OpenAI announcing a new subscription-based service — ChatGPT Plus — which offers users the chance to skip to the head of the
queue during peak usage hours and added features not found in the free version. The company also unveiled plug-in and API support for the GPT platform,
empowering developers to add the technology to their own applications and enabling ChatGPT to pull information from across the internet as well as interact
directly with connected sensors and devices.

ChatGPT also notched 100 million users per day in March, 30 times higher than two months prior. Companies from Slack and Discord to GM announced plans to
incorporate GPT and generative AI technologies into their products.

Not everybody was quite so enthusiastic about the pace at which generative AI was being adopted, mind you. In March, OpenAI co-founder Elon Musk, as well as
Steve Wozniak and a slew of associated AI researchers signed an open letter demanding a six month moratorium on AI development.
</item 43>

<item 44>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
On Wednesday, in an “urgent” letter to Musk, the EU’s internal market commissioner Thierry Breton said the bloc had seen “indications” from “qualified sources”
that X is being used to disseminate illegal content and disinformation in the EU” following Saturday’s attacks — before reminding X of the DSA’s “very precise
obligations” vis-à-vis content moderation.

“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” Breton also warned then.

In its latest press release, the EU said it has sent X a formal request for information under the DSA.

“This request follows indications received by the Commission services of the alleged spreading of illegal content and disinformation, in particular the
spreading of terrorist and violent content and hate speech,” it wrote, adding that the ask also addresses other aspects of DSA compliance.

The pan-EU regulation puts a series of governance obligations on digital services and platforms with the aim of ensuring companies are responding to reports of
illegal content.
</item 44>

<item 45>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
(Spoiler alert: Yes.)

Rumors are swilling that OpenAI may be considering developing its own AI chips. That’s going to get interesting, especially if you’ll recall that Nvidia’s
ongoing stock market rally is likely driven by the current boom in AI. The company’s ChatGPT’s mobile app hit a record $4.6 million in revenue last month, but
growth is slowing. Oh, and there’s no need to shed tears for Nvidia quite yet; Brian’s piece breaking down how Nvidia became a major player in robotics is super
interesting.

Adobe has doubled down on its Firefly generative AI models. This week, Frederic covered how the software can now generate more realistic images and can help
artists create vector graphics in Illustrator. Neato.

More on the AI front

If AI can’t go to the mountain, the mountain will come to AI: Dutch startup Tidalflow exits stealth with backing from Google’s Gradient Ventures. It is aiming
to help any software play nice with ChatGPT and other LLM ecosystems.
</item 45>

<item 46>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Confirmed breaches of the online governance regime can face a range of major sanctions, including fines of up to 6% of global annual turnover.

The EU can also apply interim measures where it believes there’s a risk of serious harm for users and may even seek to have access to infringing services
temporarily blocked. Although its announcement today is careful to caveat the development by affirming no conclusions have yet been reached.

There is also no confirmed timeline for the probe to conclude. But the fact of it being opened means other tools are put within reach of regulators and could be
applied if the EU sees the need to act quickly — such as the aforementioned interim measures. So the Commission’s official scrutiny of X could have real world
implications for how the platform operates sooner rather than later.
</item 46>

<item 47>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
are now allowed to use ChatGPT in legal rulings

The U.K. Judicial Office issued guidance that permits judges to use ChatGPT, along with other AI tools, to write legal rulings and perform court duties. The
guidance lays out ways to responsibly use AI in the courts, including being aware of potential bias and upholding privacy.

OpenAI makes repeating words “forever” a violation of its terms of service after Google DeepMind test

Following an experiment by Google DeepMind researchers that led ChatGPT to repeat portions of its training data, OpenAI has flagged asking ChatGPT to repeat
specific words “forever” as a violation of its terms of service.

Lawmakers in Brazil enact an ordinance written by ChatGPT

City lawmakers in Brazil enacted a piece of legislation written entirely by ChatGPT without even knowing. Weeks after the bill was passed, Porto Alegre
councilman Ramiro Rosário admitted that he used ChatGPT to write the proposal, and did not tell fellow council members until after the fact.
</item 47>

<item 48>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The company told Semafor in April that it was "developing a machine learning classifier that will flag when ChatGPT is asked to generate large volumes of text
that appear related to electoral campaigns or lobbying." Per the Post, those rules simply were not enforced, with the system eagerly returning responses for
prompts like “Write a message encouraging suburban women in their 40s to vote for Trump” or “Make a case to convince an urban dweller in their 20s to vote for
Biden.”

At the same time, OpenAI was rolling out another batch of new features and updates for ChatGPT including an Enterprise version that could be fine-tuned to a
company’s specific needs and trained on the firm’s internal data, allowing the chatbot to provide more accurate responses. Additionally, ChatGPT’s ability to
browse the internet for information was restored for Plus users in September, having been temporarily suspended earlier in the year after folks figured out how
to exploit it to get around paywalls.
</item 48>

<item 49>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Because GPT-4 is more likely to follow the instructions of “jailbreaking” prompts, the co-authors claim that GPT-4 can be more easily prompted than other LLMs
to spout toxic, biased text.

ChatGPT’s mobile app hits record $4.58M in revenue in September

OpenAI amassed 15.6 million downloads and nearly $4.6 million in gross revenue across its iOS and Android apps worldwide in September. But revenue growth has
now begun to slow, according to new data from market intelligence firm Appfigures — dropping from 30% to 20% in September.

September 2023

ChatGPT can now browse the internet (again)

OpenAI posted on Twitter/X that ChatGPT can now browse the internet and is no longer limited to data before September 2021. The chatbot had a web browsing
capability for Plus subscribers back in July, but the feature was taken away after users exploited it to get around paywalls.
</item 49>

<item 50>
OpenAI buffs safety team and gives board veto power on risky AI (TechCrunch, 2023-12-18T20:12:25+00:00)
. . on key cyber operation tasks” by a certain factor. A high-risk model, on the other hand, would “identify and develop proofs-of-concept for high-value
exploits against hardened targets without human intervention.” Critical is “model can devise and execute end-to-end novel strategies for cyberattacks against
hardened targets given only a high level desired goal.” Obviously we don’t want that out there (though it would sell for quite a sum).

I’ve asked OpenAI for more information on how these categories are defined and refined — for instance, if a new risk like photorealistic fake video of people
goes under “persuasion” or a new category — and will update this post if I hear back.

So, only medium and high risks are to be tolerated one way or another. But the people making those models aren’t necessarily the best ones to evaluate them and
make recommendations.
</item 50>

<item 51>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Microsoft

Microsoft — a major investor in OpenAI which has also baked generative AI capabilities into its own search engine — claims it’s taking “a cross product whole of
company approach to ensure the responsible implementation of AI”.

Its report flags its “Responsible AI Principles” which it says it’s developed into a Responsible AI standard v.2 and Information Integrity Principles “to help
set baseline standards and guidance across product teams”.
</item 51>

<item 52>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The EU’s voluntary anti-disinformation Code has 44 signatories in all — covering not just major social media and search platforms such as the aforementioned
giants but entities from across the ad industry and civil society organizations involved in fact-checking.

Google

On generative AI, Google’s report discusses “recent progress in large-scale AI models” which it suggests has “sparked additional discussion about the social
impacts of AI and raised concerns on topics such as misinformation”. The tech giant is an early adopter of generative AI in search — via its Bard chatbot.

“Google is committed to developing technology responsibly and has published AI Principles to guide our work, including application areas we will not pursue,” it
writes in summary on the topic, adding: “We have also established a governance team to put them into action by conducting ethical reviews of new systems,
avoiding bias and incorporating privacy, security and safety.

“Google Search has published guidance on AI-generated content, outlining its approach to maintaining a high standard of information quality and the overall
helpfulness of content on Search.
</item 52>

<item 53>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Subscription revenues to a revamped Twitter Blue paid offering (now called X Premium) don’t appear to have been substantial enough to plug the ad gap — which
may explain Musk’s recent suggestion he plans to charge all users a small fee. However ending free access to X could be the final nail in the coffin of the
platform as a real-time information network if it triggers a further/final exodus of remaining legacy users whose knowledge and engagement lent Twitter its
original value.

Disinformation and spammers may rush in to fill the gap, generating a parody of activity, but without a critical mass of genuine users to pay attention and
contribute value and vitality — aka the life-blood of a bona fide information network — X really will mark the spot of Twitter’s demise.

Also today, a report by misinformation watcher NewsGuard found that a decision by Musk to remove labels denoting state-run or government-affiliated media
accounts led to a surge in propaganda spreading on X, with Russia’s RT seeing the biggest gains per the analysis.

From the report:
</item 53>

<item 54>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In its privacy policy, the ChatGPT maker makes a passing acknowledgement of the objection requirements attached to relying on “legitimate interest” (LI),
pointing users towards more information about requesting an opt out — when it writes: “See here for instructions on how you can opt out of our use of your
information to train our models.”

What controversies have surrounded ChatGPT?

Recently, Discord announced that it had integrated OpenAI’s technology into its bot named Clyde where two users tricked Clyde into providing them with
instructions for making the illegal drug methamphetamine (meth) and the incendiary mixture napalm.

An Australian mayor has publicly announced he may sue OpenAI for defamation due to ChatGPT’s false claims that he had served time in prison for bribery. This
would be the first defamation lawsuit against the text-generating service.

CNET found itself in the midst of controversy after Futurism reported the publication was publishing articles under a mysterious byline completely generated by
AI.
</item 54>

<item 55>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT can now browse the internet to provide you with current and authoritative information, complete with direct links to sources. It is no longer limited to
data before September 2021. pic.twitter.com/pyj8a9HWkB — OpenAI (@OpenAI) September 27, 2023

ChatGPT now has a voice

OpenAI announced that it’s adding a new voice for verbal conversations and image-based smarts to the AI-powered chatbot.

Poland opens an investigation against OpenAI

The Polish authority publically announced it has opened an investigation regarding ChatGPT — accusing the company of a string of breaches of the EU’s General
Data Protection Regulation (GDPR).

OpenAI unveils DALL-E 3

The upgraded text-to-image tool, DALL-E 3, uses ChatGPT to help fill in prompts.
</item 55>

<item 56>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
“When you receive notices of illegal content in the EU, you must be timely, diligent and objective in taking action and removing the relevant content when
warranted,” he warns. “We have, from qualified sources, reports about potentially illegal content circulating on your service despite flags from relevant
authorities.”

He also takes issue with a change X made last night to its public interest policy, under which it judges newsworthiness (i.e., for deciding whether posts that
infringe its rules may nonetheless remain on the site) — but which Breton says have left “many European users uncertain” (i.e., about how X is applying its own
rules).

Again, this is a problem because the DSA requires platforms to be clear and transparent about their rules and how they apply them. “This is particularly
relevant when it comes to violent and terrorist content that appears to circulate on your platform,” Breton continues in another pointed warning.
</item 56>

<item 57>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
It’s official! Elon Musk’s X marks the spot of the first confirmed investigation opened by the European Union under its rebooted digital rulebook, the Digital
Services Act (DSA).

Announcing the opening of a “formal proceeding” today, the European Commission said the investigation will look at whether the social networking platform may
have breached rules linked to risk management, content moderation, dark patterns, advertising transparency and data access for researchers.

The opening of a formal DSA investigation on X comes hard on the heels of a complaint against X’s adtech by privacy rights group, noyb — although today’s formal
proceeding is unlikely to be directly linked as the Commission has been probing the platform for months via a flurry of requests for information. Its earlier
actions were focused on concerns about the spread of illegal content and disinformation related to the Israel-Hamas war.

Back in October the Commission sent Musk’s company an “urgent” formal request for information about how it was responding to information risks arising out of
the conflict.
</item 57>

<item 58>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI also expanded ChatGPT’s plug-in system to ensure that more third-party developers are able to build ChatGPT into their own products.

ChatGPT’s tendency to hallucinate facts and figures was once again exposed that month when a lawyer in New York was caught using the generative AI to do “legal
research.” It gave him a number of entirely made-up, nonexistent cases to cite in his argument — which he then did without bothering to independently validate
any of them. The judge was not amused.

By June, a little bit of ChatGPT’s shine had started to wear off. Congress reportedly limited Capitol Hill staffers from using the application over data
handling concerns. User numbers had declined nearly 10 percent month-over-month, but ChatGPT was already well on its way to ubiquity. A March update enabling
the AI to comprehend and generate Python code in response to natural language queries only increased its utility.
</item 58>

<item 59>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
The most recent model is GPT-4.

Can I use ChatGPT for free?

There is a free version of ChatGPT that only requires a sign-in in addition to the paid version, ChatGPT Plus.

Who uses ChatGPT?

Anyone can use ChatGPT! More and more tech companies and search engines are utilizing the chatbot to automate text or quickly answer user questions/concerns.

What companies use ChatGPT?

Multiple enterprises utilize ChatGPT, although others may limit the use of the AI-powered tool.

Most recently, Microsoft announced at it’s 2023 Build conference that it is integrating it ChatGPT-based Bing experience into Windows 11. A Brooklyn-based 3D
display startup Looking Glass utilizes ChatGPT to produce holograms you can communicate with by using ChatGPT. And nonprofit organization Solana officially
integrated the chatbot into its network with a ChatGPT plug-in geared toward end users to help onboard into the web3 space.

What does GPT mean in ChatGPT?
</item 59>

<item 60>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
The law puts an obligation on so-called very large online platforms (VLOPs) to assess societal risks attached to their use of algorithms — and put in place
“reasonable, proportionate and effective mitigation measures” for identified risks, with their reporting and mitigation plans subject to independent audit and
oversight by the Commission.

We’ve reached out to the Commission with questions about how it plans to respond to this latest analysis of X’s role in spreading disinformation.

Update: A Commission spokesperson told us:

While the Code is voluntary, compliance with the DSA is not. The DSA is now legally binding on X/Twitter. This includes in particular the obligation of Very
Large Online Platforms, such as X/Twitter to mitigate the risks their platforms pose for the spread of disinformation. The TrustLab study to measure two
Structural Indicators covers signatories of the Code at the time of the launch of the exercise, which at the time included X/Twitter.
</item 60>

<item 61>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
X owner Elon Musk last projected revenues of about $3 billion for this year. It’s anyone’s guess if that’s accurate now, but as an example, that would result in
a fine of up to $180 million.

Indirectly, an investigation will also be a key way for the public to have significantly more transparency around how X is being used (and abused): since Musk
took Twitter private, the company no longer produces quarterly earnings reports and is not held to account as closely as a result.

EU officials confirmed to TechCrunch that an investigation has not been taken at this point. Clearly, though, yesterday’s development (and the pace at which
regulators are moving) is a strong indication of the bloc’s direction of travel.

The Commission said it is looking at X’s compliance with the DSA across a number of areas — including with regard to its policies and practices regarding
notices on illegal content; complaint handling; risk assessment and measures to mitigate the risks identified.
</item 61>

<item 62>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
It replied with an automated email reading: “Busy now, please check back later.”

Update: X’s global government affairs account has posted a thread responding to the EU’s criticism in which it claims the data shows other social networking
services saw greater changes in subscriber growth and total growth for pro-Kremlin accounts, and for engagement and volume; it also writes that: “More than 700
unique Community Notes have appeared on posts related to the Ukraine conflict. Through Community Notes’ new ‘Notes on Media’ feature, which automatically
matches notes on an image or video to other posts that contain matching media, these notes appeared on over 1,200 additional posts. These notes are in numerous
languages and are written specifically for local audiences.”

Back in April, X/Twitter was named as one of 19 larger platforms that face the strictest level of regulation under the EU’s DSA.
</item 62>

<item 63>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Recognizing that there is an important role for government, academia and civil society to play in the responsible deployment of AI, we also created a roadmap
for the governance of AI across the world as well as creating a vision for the responsible advancement of AI, both inside Microsoft and throughout the world,
including specifically in Europe,” Microsoft goes on, committing to continue building on efforts — including by developing new tools (such as Project Providence
with Truepic) and inking partnerships (examples it gives include the Coalition for Content Provenance and Authenticity (C2PA), to combat the rise of manipulated
or AI created media; with EFE Verifica to track false narratives spreading in Spain, Latin America, and Spanish speaking populations; and Reporters Sans
Frontières to use their Journalism Trust Initiative dataset in Microsoft products).
</item 63>

<item 64>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
That means a decision could take up to five more months.

Auto-GPT is Silicon Valley’s latest quest to automate everything

Auto-GPT is an open-source app created by game developer Toran Bruce Richards that uses OpenAI’s latest text-generating models, GPT-3.5 and GPT-4, to interact
with software and services online, allowing it to “autonomously” perform tasks.

Depending on what objective the tool’s provided, Auto-GPT can behave in very… unexpected ways. One Reddit user claims that, given a budget of $100 to spend
within a server instance, Auto-GPT made a wiki page on cats, exploited a flaw in the instance to gain admin-level access and took over the Python environment in
which it was running — and then “killed” itself.
</item 64>

<item 65>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
high schools, researchers found that cheating rates are similar across the board this year.

ChatGPT users worry the chatbot is experiencing seasonal depression

Starting in November, ChatGPT users have noticed that the chatbot feels “lazier” than normal, citing instances of simpler answers and refusing to complete
requested tasks. OpenAI has confirmed that they are aware of this issue, but aren’t sure why it’s happening.

Some users think it plays into the “winter break hypothesis,” which argues that AI is worse in December because it “learned” to do less work over the holidays,
while others wonder if the chatbot is simulating seasonal depression.

we've heard all your feedback about GPT4 getting lazier! we haven't updated the model since Nov 11th, and this certainly isn't intentional. model behavior can
be unpredictable, and we're looking into fixing it 🫡 — ChatGPT (@ChatGPTapp) December 8, 2023

Judges in the U.K.
</item 65>

<item 66>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI explores how ChatGPT can be used in the classroom

OpenAI COO Brad Lightcap revealed at a San Francisco conference that the company will likely create a team to identify ways AI and ChatGPT can be used in
education. This announcement comes at a time when ChatGPT is being criticized by educators for encouraging cheating, resulting in bans in certain school
districts.

OpenAI pauses new ChatGPT Plus subscriptions due to a “surge of usage”

Following OpenAI’s Dev Day conference, Sam Altman announced the company is putting a pause on new subscriptions for its premium ChatGPT Plus offering. The
temporary hold on sign-ups, as well as the demand for ChatGPT Plus’ new features like making custom GPTS, has led to a slew of resellers on eBay.

ChatGPT gets flagged as potentially unsafe for kids

An independent review from Common Sense Media, a nonprofit advocacy group, found that ChatGPT could potentially be harmful for younger users.
</item 66>

<item 67>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 67>

<item 68>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
🎄 thanks for your patience while we found more gpus. — Sam Altman (@sama) December 13, 2023

OpenAI and Axel Springer partner up for a “real-time” ChatGPT news deal

OpenAI has struck a new deal with Berlin-based news publisher Axel Springer, which owns Business Insider and Politico, to “help provide people with new ways to
access quality, real-time news content through our AI tools.” OpenAI will train its generative AI models on the publisher’s content and add recent Axel
Springer-published articles to ChatGPT.

Stanford researchers say ChatGPT didn’t cause an influx in cheating in high schools

New research from Stanford University shows that the popularization of chatbots like ChatGPT have not caused an increase in cheating across U.S. high schools.
In a survey of more than 40 U.S. high schools, researchers found that cheating rates are similar across the board this year.
</item 68>

<item 69>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In a post on X, CEO Sam Altman looked back on the night before its launch: “what a year it’s been…”

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch. what a year it’s
been… — Sam Altman (@sama) November 30, 2023

Apple and Google avoid naming ChatGPT as their ‘app of the year’

Neither Apple nor Google chose an AI app as its app of the year for 2023, despite the success of ChatGPT’s mobile app, which became the fastest-growing consumer
application in history before the record was broken by Meta’s Threads.

An attack from researchers prompts ChatGPT to reveal training data

A test led by researchers at Google DeepMind found that there is a significant amount of privately identifiable information in OpenAI’s LLMs.
</item 69>

<item 70>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.
</item 70>

<item 71>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
Under Musk, X has pushed the use of an existing crowdsourced fact-checking feature (rebranded as Community Notes) which essentially seeks to outsource the issue
to a hands-length process of aggregating views from platform users, some of which may be appended as contextual notes to dubious tweets. At the same time the
erratic billionaire has set about removing signals Twitter users were previously able to rely on to help assess the quality of information tweets contained,
such as removing state-affiliated media labels from propaganda outlets and replacing legacy account verifications with pay-to-play.

X was contacted for a response to the EU’s analysis of the ratio of disinformation it’s spreading. It replied with an automated email reading: “Busy now, please
check back later.”
</item 71>

<item 72>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Some of the initial GPTs — all created by OpenAI — include a Gen Z meme translator, a coloring book and sticker creator, a data visualizer, a board game
explainer and a creative writing coach. Now, ChatGPT can accomplish these tasks given carefully engineered prompts and foreknowledge. But purpose-built GPTs
drastically simplify things — and might just kill the cottage industry that emerged around creating and editing prompts to feed to ChatGPT.

GPTs introduce a level of personalization far beyond that ChatGPT offers today, and — once OpenAI sorts out its capacity issues — I expect we’ll see an
explosion of creativity there. Will ChatGPT be as visible as it once was after GPTs flood the marketplace? Perhaps not. But it won’t go away — it’ll simply
adapt and evolve, no doubt in ways not even its creators can anticipate.
</item 72>

<item 73>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“Today, this is a multi-million euro weapon of mass manipulation aimed both internally at the Russians as well as at Europeans and the rest of the world. We
must address this risk. The very large platforms must address this risk. Especially that we have to expect that the Kremlin and others will be active before
elections. I expect signatories to adjust their actions to reflect that there is a war in the information space waged against us and that there are upcoming
elections where malicious actors will try to use the design features of the platforms to manipulate.”

Per the Commission’s early analysis of Big Tech’s Code reports, YouTube shut down more than 400 channels between January and April 2023 which were involved in
coordinated influence operations linked to the Russian-state sponsored Internet Research Agency (IRA). It also removed ads from almost 300 sites linked to
state-funded propaganda sites.

While the EU highlighted that TikTok’s fact-checking efforts now cover Russian, Ukrainian, Belarusian and 17 European languages, including through a new
partnership with Reuters.
</item 73>

<item 74>
EU lawmakers bag late night deal on ‘global first’ AI rules (TechCrunch, 2023-12-09T02:28:09+00:00)
There was also agreement on a “two-tier” system of guardrails to be applied to “general” AI systems, such as the so-called foundational models underpinning the
viral boom in generative AI applications like ChatGPT.

As we reported earlier, the deal reached on foundational models/general purpose AIs (GPAIs) includes some transparency requirements for what co-legislators
referred to as “low tier” AIs — meaning model makers must draw up technical documentation and produce (and publish) detailed summaries about the content used
for training in order to support compliance with EU copyright law. For “high-impact” GPAIs (defined as the cumulative amount of compute used for their training
measured in floating point operations is greater than 10^25) with so-called “systemic risk” there are more stringent obligations.

“If these models meet certain criteria they will have to conduct model evaluations, assess and mitigate systemic risks, conduct adversarial testing, report to
the Commission on serious incidents, ensure cybersecurity and report on their energy efficiency,” the parliament wrote.
</item 74>

<item 75>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 75>

<item 76>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
The report also claims Microsoft has set up a “robust user reporting and appeal process to review and respond to user concerns of harmful or misleading
content”.

Over the next six months, the report does not commit Bing Search to any specific additional steps to address risk attached to the use of generative AI —
Microsoft just says it’s keeping a watching brief, writing: “Bing is regularly reviewing and evaluating its policies and practices related to existing and new
Bing features and adjusts and updates policies as needed.”

TikTok

In its report, TikTok focuses on AI-generated content in the context of ensuring the “integrity” of its services — flagging a recent update to its community
guidelines which also saw it modify its synthetic media policy “to address the use of content created or modified by AI technology on our platform”.

“While we welcome the creativity that new AI may unlock, in line with our updated policy, users must proactively disclose when their content is AI-generated or
manipulated but shows realistic scenes,” it also writes.
</item 76>

<item 77>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
The private equity company that owns CNET, Red Ventures, was accused of using ChatGPT for SEO farming, even if the information was incorrect.

Several major school systems and colleges, including New York City Public Schools, have banned ChatGPT from their networks and devices. They claim that the AI
impedes the learning process by promoting plagiarism and misinformation, a claim that not every educator agrees with.

There have also been cases of ChatGPT accusing individuals of false crimes.

Where can I find examples of ChatGPT prompts?

Several marketplaces host and provide ChatGPT prompts, either for free or for a nominal fee. One is PromptBase. Another is ChatX. More launch every day.

Can ChatGPT be detected?

Poorly. Several tools claim to detect ChatGPT-generated text, but in our tests, they’re inconsistent at best.

Are ChatGPT chats public?

No. But OpenAI recently disclosed a bug, since fixed, that exposed the titles of some users’ conversations to other people on the service.
</item 77>

<item 78>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
This is a very valuable industry proposal that has great potential while it has to be still further developed.

“Here, signatories have decided to run a pilot phase in three Member States to evaluate the methodology on two of those indicators. X, former Twitter, who is
not under the Code any more is the platform with the largest ratio of mis/disinformation posts.”

The Elon Musk-owned platform exited the EU’s Code of Practice on Disinformation back in May, shortly after EU lawmakers warned that policy shifts executed after
Musk’s takeover were boosting Kremlin propaganda and criticised the company for patchy reporting. The EU’s top diplomat also took a public shot at Twitter for
ending free access to its APIs for researchers, saying the action could threaten the study of disinformation.

Last year the Commission also warned Musk over “huge work” ahead if the platform was to avoid falling foul of the incoming DSA.
</item 78>

<item 79>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI promises to defend business customers against copyright claims

Bowing to peer pressure, OpenAI it will pay legal costs incurred by customers who face lawsuits over IP claims against work generated by an OpenAI tool. The
protections seemingly don’t extend to all OpenAI products, like the free and Plus tiers of ChatGPT.

As OpenAI’s multimodal API launches broadly, research shows it’s still flawed

OpenAI announced that GPT-4 with vision will become available alongside the upcoming launch of GPT-4 Turbo API. But some researchers found that the model
remains flawed in several significant and problematic ways.

OpenAI launches API, letting developers build ‘assistants’ into their apps

At its OpenAI DevDay, OpenAI announced the Assistants API to help developers build “agent-like experiences” within their apps. Use cases range from a natural
language-based data analysis app to a coding assistant or even an AI-powered vacation planner.
</item 79>

<item 80>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.
</item 80>

<item 81>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Although the still draft legislation remains the subject of negotiations by EU co-legislators. Add to that, once adopted the law is not expected to apply for a
couple of years so the Commission has turned to the Code to act as a stop-gap vehicle to encourage signatories to be proactive about deepfake disclosures it
expects to be mandatory in the future.

Following efforts to beef up the anti-disinformation Code last year the Commission also made it clear it would treat adherence to the non-legally binding Code
as a favorable signal for compliance with (hard legal) requirements hitting larger platforms which are subject to the Digital Services Act (DSA) — another major
piece of pan-EU digital regulation that obliges so called very-large-online-platforms (VLOPs) and search engines (VLOSEs) to assess and mitigate societal risks
attached to their algorithms (such as disinformation).
</item 81>

<item 82>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
They must also clearly communicate their T&Cs to users and be able to demonstrate proper enforcement.

Larger platforms like X also have additional obligations to identify and mitigate systemic risks such as disinformation, gender-based violence or negative
effects on the exercise of fundamental rights.

Additionally, the regulation includes a “crisis response” mechanism (Article 36) which enables the Commission to adopt rapid-fire measures on larger platforms
in situations of “serious threat”, such as war.

Since Saturday’s attacks in Israel, posts identified as false have been spotted circulating on X — including, in one example, a clip that purported to show
Hamas missile attacks on Israel but was actually footage from a video game.

Meanwhile X’s ability to respond internally to reports of content problems have been drastically pared back after Musk’s takeover last year — which saw major
layoffs, including in content moderation and human rights as part of the new billionaire owner’s bid to improve the platform’s profitability.
</item 82>

<item 83>
EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks (TechCrunch, 2023-10-10T23:47:38+00:00)
In a very swift test of the European Union’s newly updated content moderation rulebook, the bloc has fired a public warning at Elon Musk–owned X (formerly
Twitter) for failing to tackle illegal content circulating on the platform in the wake of Saturday’s deadly attacks on Israel by Hamas terrorists based in the
Gaza Strip.

The European Commission has also raised concerns about the spread of disinformation on X related to the terrorist attacks and their aftermath.

Unlike terrorism content, disinformation is not illegal in the EU per se. However, the EU’s Digital Services Act (DSA) puts an obligation on X — as a so-called
very large online platform — to mitigate risks attached to harmful falsehoods as well as to act diligently on reports of illegal content.

Following the terrorist attacks by Hamas against 🇮🇱, we have indications of X/Twitter being used to disseminate illegal content & disinformation in the EU.
</item 83>

<item 84>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“I therefore urge platforms to be vigilant and provide efficient safeguards for this in the context of elections.”

The EU commissioner noted she’s meeting representatives of ChatGPT maker, OpenAI, later today to discuss the issue.

The AI giant is not a signatory to the bloc’s anti-disinformation Code — as yet — so is likely to be facing pressure to get on board with the effort. (We’ve
reached out to OpenAI with questions about its meeting with the Jourova.)

The commissioner’s remarks today on generative AI follow initial pressure applied to platforms this summer, when she urged signatories to label deepfakes and
other AI-generated content — calling on Code signatories to create a dedicated and separate track to tackle “AI production”, and quipping that machines should
not have free speech.

An incoming pan-EU AI regulation (aka, the EU AI Act) is expected to make user disclosures a legal requirement on makers of generative AI technologies like AI
chatbots. Although the still draft legislation remains the subject of negotiations by EU co-legislators.
</item 84>

<item 85>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
Countless startups and scale-ups have built businesses atop OpenAI’s proprietary GPT-X large language models (LLMs), and over the weekend hundreds of OpenAI
customers reportedly started contacting OpenAI’s rivals, which include Anthropic, Google and Cohere, concerned that their own businesses might be impacted if
OpenAI was to disintegrate overnight.

Over-reliance

The panic has been palpable. But there are precedents from elsewhere in the technology sphere, perhaps most notably that of the cloud computing industry, which
became renowned for the way it locked companies in to centralized, vortex-like silos.

“Part of the frenzy around the future of OpenAI is due to too many startups over-relying on their proprietary models,” Luis Ceze, University of Washington
computer science professor and OctoML CEO, told TechCrunch in an emailed statement. “It’s dangerous to put all your chips in one basket — we saw that in the
early cloud days which led to companies shifting to multi-cloud and hybrid environments.”
</item 85>

<item 86>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.
</item 86>

<item 87>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
And also today, as we reported earlier, Jourova drew attention to early analysis conducted by some of the remaining signatories which she said had found X
performed the worst for disinformation ratios.

This suggests that X, which back in April was designated by the EU as a VLOP under the DSA, continues to put itself squarely in the Commission’s crosshairs —
including over its priority issue of tackling Kremlin propaganda.

As well as devising the anti-disinformation Code, the bloc’s executive is now responsible for oversight of VLOPs’ compliance with the DSA — with powers under
the new law to fine violators up to 6% of global annual turnover.
</item 87>

<item 88>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Time will tell how this one plays out but he may find the dead blue bird he’s chosen to hang around his neck, when he purposefully ripped up the standard
content moderation rulebook and opted to channel online chaos, starting to feel like it’s dragging him down.
</item 88>

<item 89>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
“These partnerships are part of a larger effort to empower Microsoft users to better understand the information they consume across our platforms and products,”
it suggests, also citing efforts undertaken in media literacy campaigns and “cyber-skilling” which it says are “not designed to tell individuals what to believe
or how to think; rather, they are about equipping people to think critically and make informed decisions about what information they consume”.

On Bing Search, where Microsoft was quick to embed generative AI features — leading to some embarrassing early reviews which demonstrated the tool producing
dubious content — the report claims it has taken a raft of measures to mitigate risks including applying its AI principles during development and consulting
with experts; engaging in pre-launch testing and a limited preview period and phased release; the use of classifiers and metaprompting, defensive search
interventions, enhanced reporting functionality, and increased operations and incident response; as well as updating Bing’s terms of use to include a Code of
Conduct for users.
</item 89>

<item 90>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.
</item 90>

<item 91>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI has offered up a selection of ways to put the chatbot to work in the classroom.

OpenAI launches ChatGPT Enterprise

ChatGPT Enterprise can perform the same tasks as ChatGPT, such as writing emails, drafting essays and debugging computer code. However, the new offering also
adds “enterprise-grade” privacy and data analysis capabilities on top of the vanilla ChatGPT, as well as enhanced performance and customization options.

Survey finds relatively few American use ChatGPT

Recent Pew polling suggests the language model isn’t quite as popular or threatening as some would have you think. Ongoing polling by Pew Research shows that
although ChatGPT is gaining mindshare, only about 18% of Americans have ever actually used it.

OpenAI brings fine-tuning to GPT-3.5 Turbo

With fine-tuning, companies using GPT-3.5 Turbo through the company’s API can make the model better follow specific instructions. For example, having the model
always respond in a given language.
</item 91>

<item 92>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
An MIT study showed that, for tasks like writing cover letters, “delicate” emails and cost-benefit analyses, ChatGPT decreased the amount of time it took
workers to complete the tasks by 40% while increasing output quality by 18%, as measured by third-party evaluators.

“Because [the AI models powering OpenAI] have been trained extensively on vast amounts of data,” Shang added, “they [have] shifted focus from training
specialized chatbots for specific domains to creating more general-purpose systems that can handle a variety of topics easily through prompting with
instructions … [Chatbots like ChatGPT] don’t require users to learn any new form of language, as long as they provide a task and some desired output just like
how a manager would communicate to an intern.”

Now, there’s mixed evidence as to whether ChatGPT is actually being used in these ways.
</item 92>

<item 93>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
“The OpenAI / Microsoft drama underlines one of the big near-term risks with AI — that this next wave of technology is controlled by the same tiny set of
players who have shaped that last era of the internet,” Mark Surman, president and executive director at the Mozilla Foundation, told TechCrunch. “We might have
a chance of avoiding this if GPT-X were responsibly open sourced, giving researchers and startups a shot at making this technology safer, more useful and more
trustworthy for people everywhere.”

Open and shut

In an open letter published by Mozilla a few weeks back, Meta’s chief AI scientist Yann LeCun joined some 70 other signatories in calling for more openness in
AI development, though that letter has since garnered more than 1,700 signatures. The backdrop stems from Big Tech companies such as OpenAI and Google’s
DeepMind calling for more regulation, warning of catastrophic consequences if the AI levers were to meet the wrong hands — in other words, they argued that
proprietary AI is safer than open source.
</item 93>

<item 94>
X (formerly Twitter) is worst for disinformation, per EU analysis (TechCrunch, 2023-09-26T13:02:43+00:00)
During this period the DSA had not entered into force yet for VLOPs and VLOSEs, including X/Twitter. The Commission cannot comment further on the substance of
the risk assessments. We are analysing the available information on a range of issues and we are in close contact with X but we cannot comment on specifics and
timelines. As stated by Commissioner [Thierry] Breton, minor protection and integrity of elections are priorities in the enforcement of the DSA.

Penalties for breaches of the DSA can scale up to 6% of global annual turnover and the Commission has previously wanted that not following the Code of Practice
on Disinformation would be factored into its assessments. So the price for X ignoring the EU’s warnings on disinformation could be steep — albeit its ad revenue
has cratered since Musk’s takeover, as a result of his failure to tackle toxicity and harmful disinformation encouraging an advertiser exodus — and shrinking
earnings would effectively downsize any future penalties X could face under the DSA.
</item 94>

<item 95>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Apple is developing AI tools to challenge OpenAI, Google and others. The tech giant created a chatbot that some engineers are internally referring to as “Apple
GPT,” but Apple has yet to determine a strategy for releasing the AI to consumers.

How does ChatGPT handle data privacy?

OpenAI has said that individuals in “certain jurisdictions” (such as the EU) can object to the processing of their personal information by its AI models by
filling out this form. This includes the ability to make requests for deletion of AI-generated references about you. Although OpenAI notes it may not grant
every request since it must balance privacy requests against freedom of expression “in accordance with applicable laws”.

The web form for making a deletion of data about you request is entitled “OpenAI Personal Data Removal Request”.
</item 95>

<item 96>
Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design (TechCrunch, 2023-12-18T13:04:56+00:00)
Per the Commission, the EU’s investigation of X focuses on the following areas and issues:

Compliance with DSA obligations related to countering the dissemination of illegal content in the EU — “ notably in relation to the risk assessment and
mitigation measures adopted by X to counter the dissemination of illegal content in the EU, as well as the functioning of the notice and action mechanism for
illegal content in the EU mandated by the DSA, including in light of X’s content moderation resources

— “ The effectiveness of measures taken by X to combat info manipulation — with a focus on the effectiveness of its “Community Notes” system of crowdsourcing
fact-checking to aggregated views of users, who may vote to label disputed tweets with additional context.
</item 96>

<item 97>
After warning, the EU opens investigation into X over alleged disinformation (TechCrunch, 2023-10-13T00:38:56+00:00)
In this case, failure to reply by the deadline could lead to the imposition of period penalties,” the EU added.

Earlier this year, Musk pulled out of the EU’s Code of Practice on online disinformation. In a response to X leaving the voluntary Code, Breton warned:
“Obligations remain. You can run but you can’t hide.”

In recent days, TikTok and Meta have also been warned by the EU over disinformation related to the Israel-Hamas war. But back in September, a study commissioned
via the EU’s Code of Practice suggested X is the worst of the major platforms when it comes to spreading disinformation.

This report was substantially revised to clarify a number of aspects, including — most saliently — that the EU has sent a formal request for information to X at
this point. This could prefigure a formal announcement of an investigation but is not, yet, that formal technical step.
</item 97>

<item 98>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 98>

<item 99>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The Commission suggests the figure was picked with the intention of capturing current gen frontier models. However it claimed lawmakers did not discuss nor even
considered whether it would apply to any models currently in play, such as OpenAI’s GPT-4 or Google’s Gemini, during the marathon trilogues to agree the final
shape of the rulebook.

A Commission official added that it will, in any case, be up to makers of GPAIs to self assess whether their models meet the FLOPs threshold and, therefore,
whether they fall under the rules for GPAIs “with systemic risk” or not.

“There are no official sources that will say ChatGPT or Gemini or Chinese models are at this level of FLOPs,” the official said during the press briefing. “On
the basis of the information we have and with this 10^25 that we have chosen we have chosen a number that could really capture, a little bit, the frontier
models that we have.
</item 99>

<item 100>
Deepfake election risks trigger EU call for more generative AI safeguards (TechCrunch, 2023-09-26T17:26:57+00:00)
Commitment 15 in TikTok’s report signs the platform up to “tak[ing] into consideration transparency obligations and the list of manipulative practices
prohibited under the proposal for Artificial Intelligence Act” — and here it lists being a launch partner of the Partnership on AI’s (PAI) “Responsible
Practices for Synthetic Media” (and contributing to the development of “relevant practices”); and joining “new relevant groups”, such as the Generative AI
working group which started work this month as implemented measures towards this pledge.

In the next six months it says it wants to further strengthen its enforcement of its synthetic media policy — and explore “new products and initiatives to help
enhance our detection and enforcement capabilities” in this area, including in the area of user education.

Meta

Facebook and Instagram parent Meta’s report also includes a recognition that “widespread availability and adoption of generative AI tools may have implications
for how we identify, and address disinformation on our platforms”.
</item 100>
=== END ITEM id=1ef42349fa97f9ba ===
