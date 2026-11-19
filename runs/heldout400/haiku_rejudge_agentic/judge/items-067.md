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

This file has 3 items.

=== ITEM id=9e3f8fd26f9606c4 ===
Gold sentences and facts:
Sentence 0: But it was ChatGPT’s explosion into public consciousness that thrust Altman into the spotlight as a face of generative AI — technology that can produce novel imagery, passages of text and other media.
  Fact 0.0: ChatGPT's explosion into public consciousness thrust Altman into the spotlight.
  Fact 0.1: Altman became a face of generative AI because of ChatGPT's explosion into public consciousness.
  Fact 0.2: Generative AI is technology that can produce novel imagery.
  Fact 0.3: Generative AI is technology that can produce passages of text.
  Fact 0.4: Generative AI is technology that can produce other media.
Sentence 1: This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing FTX fallout.
  Fact 1.0: This week's coverage includes Sam Altman backing a teen's AI startup.
  Fact 1.1: This week's coverage includes Google's hardware event.
  Fact 1.2: This week's coverage includes first impressions of the Pixel 8 Pro.
  Fact 1.3: This week's coverage includes Flexport drama.
  Fact 1.4: This week's coverage includes the ongoing FTX fallout.
Sentence 2: Based on the board’s language and the way these giant tech companies work, this is the prevailing theory floating around right now.
  Fact 2.0: The prevailing theory is based on the board's language.
  Fact 2.1: The prevailing theory is based on the way giant tech companies work.
  Fact 2.2: The prevailing theory is floating around right now.

Retrieved items:
<item 1>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> The New York Times
- One year later, ChatGPT is still alive and kicking -MENTIONS-> disinformation
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Amazon Q
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Stella Biderman
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Booz Allen Hamilton

Entities: One year later, ChatGPT is still alive and kicking, security assessments, Steve Wozniak, open letter calling for the immediate pause of large-scale AI
experiments, large-scale AI experiments, OpenAI's models, AI-powered chatbots, cookbook collection, company's proprietary codebases, developers, ChatGPT abuse,
Signing an open letter calling for the immediate pause of large-scale AI experiments, ChatGPT, Apple, GPTs
</item 1>

<item 2>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
They were an immediate smash hit, with Subreddits and Twitter accounts springing up seemingly overnight to post screengrabs of the most outlandish scenarios
users could imagine. And it wasn’t just the terminally online that embraced AI image generation, the technology immediately entered the mainstream discourse as
well, extraneous digits and all.

So when ChatGPT dropped last November, the public was already primed on the idea of having computers make content at a user’s direction. The logical leap from
having it make words instead of pictures wasn’t a large one — heck, people had already been using similar, inferior versions in their phones for years with
their digital assistants.

Q1: [Hyping intensifies]

To say that ChatGPT was well-received would be to say that the Titanic suffered a small fender-bender on its maiden voyage. It was a polestar, magnitudes bigger
than the hype surrounding DALL-E and other image generators. People flat out lost their minds over the new AI and its CEO, Sam Altman.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> DALL-E
- Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs -MENTIONS-> DALL-E
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT

Entities: DALL-E, ChatGPT, AI image generation, Titanic, How OpenAI's ChatGPT has changed the world in just a year, OpenAI, Sam Altman, Twitter, Engadget
</item 2>

<item 3>
Graph query row:
article: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier
published_on: 2023-10-07T20:15:26+00:00
publisher: TechCrunch
matched_entity: ChatGPT
related_facts: ['SBF Trial', 'Google’s hardware event', 'Pixel 8 Pro', 'Flexport drama', 'FTX fallout', 'Gmail', 'ad-free subscription plan', 'Induced AI',
'Manish', 'Google’s annual hardware event', 'Pixel Fold', 'Pixel Buds', 'Pixel Watch 2', 'teens’ startup', 'ad-free tier', 'Pixel 7 Pro', 'recent reporting on
Flexport', 'Ingrid', 'Recruiter', 'LinkedIn Learning coach', 'AI-powered tool for marketing campaigns', 'TikTok’s ad-free plan', 'X users’ daily posts and
content generation', 'X users’ daily posts and content generation figures', 'Pixel 8', 'IronNet', 'Keith Alexander', 'NSA', 'going public', 'blockchain',
'Equity', 'At One Ventures', 'Section 32', 'Greylock', 'Fearless Fund', 'grants', 'Black women founders', 'Found', 'Esther Rodriguez-Villegas', 'Acurable',
'patient-friendly wearable devices', 'respiratory conditions', 'epilepsy', 'Chain Reaction', 'Jacquelyn', 'Southern District of New York courthouse',
'Bankman-Fried', 'SBF and FTX trial', 'Breathe Battery Technologies', 'software', 'lithium-ion battery', 'faster charging speeds', 'greater longevity', 'Anna',
'10 investors', 'survey of 10 investors', 'oligopoly dynamics', 'first-mover advantages', 'proprietary data', 'Linda Yaccarino', 'Darrell', 'ChatGPT', 'AI
tools', 'LinkedIn', 'Rainforest', 'Virginia', 'TechCrunch+', 'Tim', 'the defense', 'Bard', 'X', 'Christine', 'AI', 'spam', 'Elon Musk', 'startups', 'FTX',
'Google', 'Rebecca', 'the prosecution', 'TikTok', 'cybersecurity', 'CNBC', 'venture funding', 'sleep apnea', 'Alex', 'Sam Altman', 'Google Assistant', 'Android
14', 'U.S.', 'fraud', 'Dave Clark', 'Flexport', 'Amazon']
</item 3>

<item 4>
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
It has been a whirlwind four days for OpenAI, the generative AI poster child behind the smash hit ChatGPT.

Seemingly out of nowhere, the OpenAI board ousted CEO and co-founder Sam Altman and demoted president and co-founder Greg Brockman, who subsequently resigned,
paving the way for what looked like a mutiny by staff insisting the founders be reinstated post-haste. By then, Microsoft had already hired Altman and Brockman
to head up a new internal AI unit, though, as things transpired, nothing had actually been signed yet, with rumors suggesting that the ousted leaders might
actually return to OpenAI after all — in some capacity, at least.

The situation remains fluid, and any number of potential outcomes still remain on the table. But the whole debacle has shone a spotlight on the forces that
control the burgeoning AI revolution, leading many to question what happens if you go all-in on a centralized proprietary player, and what happens if things
then go belly-up?

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Microsoft, new internal AI unit, open AI movement, AI revolution, How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement, Meta,
TechCrunch, OpenAI, OpenAI board, Greg Brockman, Sam Altman, generative AI
</item 4>

<item 5>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
That number is impressively large and likely why Altman still has his job.

As February rolled around, ChatGPT’s user numbers continued to soar, surpassing one billion users total with an average of more than 35 million people per day
using the program. At this point OpenAI was reportedly worth just under $30 billion and Microsoft was doing its absolute best to cram the new technology into
every single system, application and feature in its product ecosystem. ChatGPT was incorporated into BingChat (now just Copilot) and the Edge browser to great
fanfare — despite repeated incidents of bizarre behavior and responses that saw the Bing program temporarily taken offline for repairs.

Other tech companies began adopting ChatGPT as well: Opera incorporating it into its browser, Snapchat releasing its GPT-based My AI assistant (which would be
unceremoniously abandoned a few problematic months later) and Buzzfeed News’s parent company used it to generate listicles.

Graph facts:
- How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement -MENTIONS-> Altman
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> Altman
- OpenAI's ex-chairman accuses board of going rogue in firing Altman: 'Sam and I are shocked and saddened by what the board did' -MENTIONS-> Altman
- WTF is going on at OpenAI? We have theories -MENTIONS-> Altman
- Altman -SAID-> copyright problems

Entities: Altman, ChatGPT, Opera, Microsoft, BingChat, Edge browser, Snapchat, My AI assistant, Buzzfeed News, Buzzfeed News’s parent company, How OpenAI's
ChatGPT has changed the world in just a year, OpenAI, Engadget
</item 5>

<item 6>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
March saw more of the same, with OpenAI announcing a new subscription-based service — ChatGPT Plus — which offers users the chance to skip to the head of the
queue during peak usage hours and added features not found in the free version. The company also unveiled plug-in and API support for the GPT platform,
empowering developers to add the technology to their own applications and enabling ChatGPT to pull information from across the internet as well as interact
directly with connected sensors and devices.

ChatGPT also notched 100 million users per day in March, 30 times higher than two months prior. Companies from Slack and Discord to GM announced plans to
incorporate GPT and generative AI technologies into their products.

Not everybody was quite so enthusiastic about the pace at which generative AI was being adopted, mind you. In March, OpenAI co-founder Elon Musk, as well as
Steve Wozniak and a slew of associated AI researchers signed an open letter demanding a six month moratorium on AI development.

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> Steve Wozniak
- Steve Wozniak -PARTICIPATED_IN-> Open letter calling for a pause in AI training
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Steve Wozniak
- Steve Wozniak -PARTICIPATED_IN-> Signing an open letter calling for the immediate pause of large-scale AI experiments
- Steve Wozniak -FOUNDED-> Apple

Entities: Steve Wozniak, ChatGPT Plus, ChatGPT, GPT platform, AI researchers, Signing an open letter demanding a six month moratorium on AI development, Six
month moratorium on AI development, How OpenAI's ChatGPT has changed the world in just a year, Slack, Elon Musk, GM, OpenAI, Discord, generative AI, Engadget
</item 6>

<item 7>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
People flat out lost their minds over the new AI and its CEO, Sam Altman. Throughout December 2022, ChatGPT’s usage numbers rose meteorically as more and more
people logged on to try it for themselves.

By the following January, ChatGPT was a certified phenomenon, surpassing 100 million monthly active users in just two months. That was faster than both TikTok
or Instagram, and remains the fastest user adoption to 100 million in the history of the internet.

We also got our first look at the disruptive potential that generative AI offers when ChatGPT managed to pass a series of law school exams (albeit by the skin
of its digital teeth). Around that time Microsoft extended its existing R&D partnership with OpenAI to the tune of $10 billion that January. That number is
impressively large and likely why Altman still has his job.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Microsoft, a series of law school exams, extension of existing R&D partnership, How OpenAI's ChatGPT has changed the world in just a year,
OpenAI, Instagram, TikTok, Sam Altman, generative AI, Engadget
</item 7>

<item 8>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, How OpenAI's ChatGPT has changed the world in just a year, Dave Willner, ChatGPT Android app, FTC investigation into OpenAI, consumer
protection laws, user data leak, Getty Images, Sara Silverman, Book2 dataset, Getty Images lawsuit against Stability AI, Sara Silverman's lawsuit against
OpenAI, copyright, FTC, Stable Diffusion
</item 8>

<item 9>
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

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Clyde, methamphetamine, napalm, Australian mayor, CNET, Futurism, bribery,
TechCrunch, defamation, AI, OpenAI, Discord
</item 9>

<item 10>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.

Graph facts:
- Founders, are events useful? -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT

Entities: LLaMA family, Fortune 500 companies, ChatGPT, ChatGPT: Everything you need to know about the AI-powered chatbot, Meta, TechCrunch, OpenAI, AI
startups, Sam Altman
</item 10>
=== END ITEM id=9e3f8fd26f9606c4 ===

=== ITEM id=c8d2ac9ac13f4a6e ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: This will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy.
  Fact 1.2: Epic did this on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google "siphons off" news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case says this is done through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.

Graph facts:
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Epic Games’ antitrust battle with Google, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic
Games, Sideloading, Security risks, Off-the-record chats, Chat logs and documents provided to the court, Play Store access and payment, TechCrunch, Google,
competition, Play Store, Fortnite
</item 1>

<item 2>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
First, Epic announced it was bypassing Apple and Google’s app store fees. It deployed a hotfix update to Fortnite without either company’s knowledge, letting
you purchase V-Bucks directly through its own payment processing option at a discount. Apple and Google almost immediately reacted by kicking Fortnite off their
app stores for breaking the rules.

Then: surprise! Epic was ready and waiting with two lawsuits and an attack ad, depicting a Fortnite hero throwing a unicorn-llama hammer into a giant screen
reminiscent of Apple’s famous “1984” Macintosh ad.

It was a striking publicity blitz followed by a lot of slow-moving court proceedings. While the Apple lawsuit went to court in 2021, the Google one was delayed
again and again. A ruling came down for the Apple trial that September, and it was mostly decided in Apple’s favor, though both parties are waiting for the
Supreme Court to potentially weigh in. Meanwhile, the machinations for its fight against Google continued, and now…

It is time for trial number two.

Graph facts:
- 200+ of the best deals from Amazon's Cyber Monday sale -MENTIONS-> Apple
- ASX set to drop as Wall Street’s September slump deepens -MENTIONS-> Apple
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Apple
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Apple
- Apple -MENTIONS-> Apple's policy for third-party apps on its devices

Entities: Apple, Supreme Court, Epic, Google, V-Bucks, Macintosh, Apple trial, Google lawsuit, Google trial, Apple's “1984” Macintosh ad, Epic v. Google,
explained, Fortnite, The Verge
</item 2>

<item 3>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The future of Google’s app store is at stake in a lawsuit by Fortnite publisher Epic Games. Epic sued Google in 2020 after a fight over in-app purchase fees,
claiming the Android operating system’s Google Play Store constituted an unlawful monopoly — while Google says its demands would damage Android’s ability to
offer a secure user experience and compete with Apple’s iOS. Follow along with updates here.

Bornstein showed jurors charts of Google’s fat app profit margins (70 percent on $12 billion in revenue a year, says Epic) and pointed out several ugly-seeming
ways Google has allegedly attempted to keep anyone from taking that money away — like paying game developers not to build their own app stores or standalone app
launchers like Epic did with Fortnite.

“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying,
‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Google Play Store, Epic Games, Apple, Google, The Epic v. Google trial may come down to simple v. complicated, unlawful monopoly, Bornstein,
Epic sued Google, iOS, Epic v. Google trial, Fortnite, The Verge
</item 3>

<item 4>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
The competitive landscape with other app stores, OEMs

The tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does
the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app
stores like Samsung’s Galaxy Store and Amazon’s Appstore.

Epic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are
still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment
manufacturers) have to agree to in order to use Android.

Epic will also bring up “Anti-Fragmentation Agreements” (AFAs) that prevent the OEMs from modifying Android to allow for frictionless downloading of apps
outside of Google Play.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Google Play
- The Epic v. Google trial may come down to simple v. complicated -MENTIONS-> Google Play
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google Play
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Google Play
- Beeper Mini is back in operation after Apple’s attempt to shut it down -MENTIONS-> Google Play

Entities: Epic Games’ antitrust battle with Google, Google Play, App Store, Android, Galaxy Store, Amazon Appstore, Android OEMs, Google apps,
Anti-Fragmentation Agreements, Ninth Circuit, Ninth Circuit ruling in the Apple case, App store competition, What to know about Fortnite maker Epic Games’
antitrust battle with Google, starting today, Epic Games, Apple
</item 4>

<item 5>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Google Play
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Google Play
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Google Play
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Google Play
- Beeper Mini is back in operation after Apple’s attempt to shut it down -MENTIONS-> Google Play

Entities: Google Play, Activision, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, app store and payment processing fees, Epic v.
Google trial, competition, The Verge
</item 5>

<item 6>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.

Graph facts:
- Alphabet -ACCUSED_OF-> Sherman Act
- Alphabet -ACCUSED_OF-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. antitrust law
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Sherman Act
- Epic v. Google, explained -MENTIONS-> User Choice Billing

Entities: U.S. antitrust law, Sherman Act, User Choice Billing, Google first-party billing system, Epic Games’ antitrust battle with Google, California’s
anticompetitive laws, Cartwright Act, Unfair Competition Law, distribution of apps to Android users, payment processing solutions for content inside Android
apps, Android, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, Epic Games, TechCrunch, Google
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
But if you do want to see a vision of a future that Epic would accept, look no further than this letter it sent to state attorneys general — laying out a
settlement that would basically prohibit Google from doing the anticompetitive things that Epic claims it did.

There’s also this tweet from Sweeney:

Google wouldn’t tell The Verge if it’s offered Epic any kind of settlement yet.

Why if I just want the dirty laundry on both these companies? You promised dirt...

Oh, it could get good.

The allegations made before trial suggest Google was playing loads of dirty tricks with names like “Project Hug,” “Project Agave,” and “Project Banyan”, with
Epic claiming it’s seen evidence Google was paying off game developers and phone makers not to abandon its app store — to the tune of hundreds of millions of
dollars. Google allegedly even toyed with the idea of buying Epic to keep Fortnite from inspiring other game developers to abandon ship.

Plus, there’s evidence Google deleted potentially incriminating messages to keep them out of a court’s hands.

Graph facts:
- Tim Sweeney on Epic’s victory royale over Google -MENTIONS-> Project Hug
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Project Hug
- Tim Sweeney -SAID-> Project Hug
- Project Hug -PART_OF-> Google Games Velocity Program
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Epic

Entities: Project Hug, Epic, Google, Epic v. Google, explained, Sweeney, state attorneys general, Project Agave, Project Banyan, game developers, phone makers,
Google's app store, settlement, anticompetitive practices, paying game developers and phone makers not to abandon Google's app store, deleting potentially
incriminating messages
</item 7>

<item 8>
Graph query row:
article: 200+ of the best deals from Amazon's Cyber Monday sale
publisher: Mashable
relatedEntities: ['Echo', 'Cyber Weekend', 'Echo Show', 'smart screen', 'smart lights', 'Audible Premium Plus', 'Echo smart displays', 'Echo smart speakers',
'Fire TVs', 'MGM+', 'Liquid Retina display', 'A14 Bionic chip', 'USB-C charging', 'Microsoft Surface Laptop Go 3', 'Surface Go 2', 'Go 3', 'QuietComfort 45s',
"Amazon's Cyber Monday sale", 'META50', "Mashable's daily Deals newsletter", 'Terms of Use', 'Tablet deals', 'Gaming keyboards', 'VR headsets', 'Vacuum deals',
'Cordless vacuums', 'Robot vacuums and vacuum/mop hybrids', 'Kitchen deals', 'Pizza ovens', 'Sparkling water makers', 'Toaster ovens', 'Amazon device and
service deals', 'Blink', 'Chromecast', 'Google Nest', 'Fitness deals', 'Smartwatches and fitness trackers', 'Apple deals', 'Beauty tech deals', 'Drone deals',
'Camera deals', 'GoPro', 'Lego deals', 'Gift card deals', 'Mac', 'Microsoft', 'Apple', 'AirPods', 'AirPods Max', 'Echo devices', 'Alexa', 'privacy policy',
'iPad', 'Eero', 'Meta Quest 2', 'Apple Watch', 'Smart Home Deals', 'Speaker Deals', 'Bose', 'Earbuds', 'MacBooks', 'MacBook Air', 'ANC', 'Echo Buds', 'MacBook',
'headphones', 'Fire TV', 'Black Friday', 'Kindle', 'Prime', 'Gaming mice', 'Black Friday sale', 'Intel', 'Amazon', 'Cyber Monday sale', 'Intel Core i5
processor', 'Fire tablets', 'gaming deals', 'Amazon credit', 'gaming headsets', 'Apple Watches']
passage: 200+ of the best deals from Amazon's Cyber Monday sale (Mashable, 2023-11-27T08:45:59+00:00)
Table of Contents Table of Contents Echo, Fire TV, and Kindle deals Apple deals TV deals Laptop deals Headphone and earbud deals Tablet deals Gaming deals
Speaker deals Vacuum deals Kitchen deals Smart home deals Fitness deals Beauty tech deals Drone deals Camera deals Lego deals Gift card deals

UPDATE: Nov. 27, 2023, 5:00 a.m. EST This post has been updated with all of the latest Cyber Monday deals available at Amazon.

Amazon is dragging out the year's biggest shopping holiday(s) into 11 days of deals.

The retail giant began its Black Friday sale in the early morning of Friday, Nov. 17 (a week ahead of schedule) and was on top of making the switch to Cyber
Monday language in the wee hours of Saturday, Nov. 25. Official Cyber Monday mode, which is currently on through Monday, Nov. 27, includes both a ton of deals
carried over from Black Friday plus some new ones.
</item 8>

<item 9>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.

Graph facts:
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- European football: Eintracht Frankfurt hit Bayern for five in famous thrashing -MENTIONS-> Android

Entities: Android, Galaxy Store, What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today, OnePlus, Epic Games app, Google
Play Store, Android app developers, Project Hug, Google Games Velocity Program, antitrust battle, agreement with OnePlus, deal between Google and Samsung,
Google developer agreements, anti-competitive practices, Epic Games
</item 9>

<item 10>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.

Graph facts:
- What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today -MENTIONS-> Android
- ChatGPT’s mobile app hits record $4.58M in revenue in September -MENTIONS-> Android
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> Android
- The best Black Friday deals you can get for under $25 -MENTIONS-> Android
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -MENTIONS-> Android

Entities: Android, Apple, Epic, Google, The Epic v. Google trial may come down to simple v. complicated, Pomerantz, Project Liberty, Android sideloading flow,
app store fees, Google policy, app stores, Epic v. Google trial, Fortnite, The Verge
</item 10>
=== END ITEM id=c8d2ac9ac13f4a6e ===

=== ITEM id=cbc3b621d78a31c1 ===
Gold sentences and facts:
Sentence 0: Because Flexport makes money by charging fees for the transportation of goods, the company's business was getting hammered.
  Fact 0.0: Flexport makes money by charging fees for the transportation of goods.
  Fact 0.1: Flexport's business was getting hammered.
Sentence 1: Petersen has spent the past month cutting costs, including laying off about 20% of its workers, or about 600 people.
  Fact 1.0: Petersen has spent the past month cutting costs.
  Fact 1.1: Petersen has laid off about 20% of Flexport's workers.
  Fact 1.2: Petersen's layoffs were about 600 people.
Sentence 2: Turmoil at Flexport: Dave Clark, the former Amazon executive who was ousted as CEO of Flexport just a year into the job, fired back at its founder and board, calling recent reporting on the logistics company “deeply concerning.” Clark made the comments Monday in a lengthy post on social media site X following a report from CNBC that provided new information about his last days at Flexport, a freight forwarding and customs brokerage startup valued at $8 billion.
  Fact 2.0: There was turmoil at Flexport.
  Fact 2.1: Dave Clark is a former Amazon executive.
  Fact 2.2: Dave Clark was ousted as CEO of Flexport just a year into the job.
  Fact 2.3: Dave Clark fired back at Flexport's founder and board.
  Fact 2.4: Dave Clark called recent reporting on Flexport 'deeply concerning'.
  Fact 2.5: Dave Clark made the comments on Monday in a lengthy post on the social media site X.
  Fact 2.6: Dave Clark's post followed a CNBC report.
  Fact 2.7: The CNBC report provided new information about Dave Clark's last days at Flexport.
  Fact 2.8: Flexport is a freight forwarding and customs brokerage startup.
  Fact 2.9: Flexport is valued at $8 billion.

Retrieved items:
<item 1>
Graph query row:
entity: Flexport drama
report: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
page: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
passage: Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Hiya, folks, and welcome to Week in Review (WiR), TechCrunch’s digest of the past week in tech news. It’s TC’s column that highlights the major stories over the
past few days, and — we humbly submit — it’s a darn useful resource for folks on the go.

This week, we cover Sam Altman backing a teen’s AI startup, Google’s hardware event (and first impressions of the Pixel 8 Pro), Flexport drama, and the ongoing
FTX fallout. Also on the agenda: Gmail’s harsher rules to prevent spam, TikTok testing an ad-free subscription plan, and LinkedIn going big on AI tools. And
that’s not all.

If you haven’t, sign up here to get WiR in your inbox every Saturday. And if you have, our thanks. Now, let’s get on with the news.
</item 1>

<item 2>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
VinFast

VinFast has not shared how many employees have been cut, but a LinkedIn post from a former employee said “nearly 35 roles” were affected. Announced on February
6.

Dell

Announced February 6, impacting 6,650 people, or 5% of worldwide workforce.

Getaround

Announced February 2, 10% of staff — about 42 employees.

Pinterest

Announced February 2, 150 employees impacted. This is the second job-cutting move within weeks of the first round in December 2022.

Rivian

Announced on February 1, cutting 6% of its workforce for the second time in less than a year.

January 2023

SoFi Technologies

Announced on January 31, cutting 65 jobs, or about 5% of its 1,300-person workforce. First reported by The Wall Street Journal.

NetApp

Announced on January 31, impacting 8% of its staff — about 960 people.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> The Wall Street Journal
- report on Thursday -PUBLISHED_BY-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy

Entities: The Wall Street Journal, tech layoffs, A comprehensive list of 2023 tech layoffs, VinFast, Getaround, Pinterest, SoFi Technologies, NetApp, VinFast
layoffs, Dell layoffs, Getaround layoffs, Pinterest layoffs, Rivian layoffs, SoFi Technologies layoffs, NetApp layoffs
</item 2>

<item 3>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
The company is now facing an unfair labor practice complaint, with workers alleging the layoffs are retaliation for unionizing.

Chewy

Laid off more than 200 employees November 14. The cuts impact roles in HR, recruiting, data, business intelligence, plus directors and higher managers.

Amazon

Is cutting “just over 180” roles in its gaming division, VP Christoph Hartmann wrote in an email to employees November 13.

Carta

Is reportedly cutting more of its workforce. The number of positions affected are unknown, but it would be the unicorn startup’s third layoff round this year.

Pico

Is cutting a “few hundred” roles, a person familiar with the matter told TechCrunch November 9. ByteDance’s Oculus challenger held an internal meeting
announcing it is halting its expansion this week.

Cruise

Started laying off contingent workers who supported driverless operations November 9. Those layoffs came after the company recalled 950 robotaxis and lost key
commercial permits to operate in California.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Chewy, Carta, Pico, Christoph Hartmann, Oculus, unfair labor practice complaint, unionizing, driverless
operations, Chewy layoffs, Amazon gaming division cuts, Pico cuts, Cruise contingent-worker layoffs, Pico internal meeting, Cruise recall of 950 robotaxis
</item 3>

<item 4>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Groupon

Impacting another 500 employees announced on January 31. The company said this new set of layoffs will be spread across the first two quarters of 2023.

Impossible Foods

Reportedly affecting 20% of its staff, over 100 employees, Bloomberg reported first.

PayPal

Announced on January 30, about 2,000 full-time employees, or 7% of its workforce, were affected.

Arrival

Announced on January 30, with a newly appointed CEO, slashing 50% of its workforce — 800 employees globally.

Waymo

The self-driving technology unit under Alphabet quietly laid off workers on January 24, according to The Information and several posts on LinkedIn and Blind.
It’s not yet clear how many of Waymo’s staff will be affected.

Spotify

Announced on January 23, impacting around 6% of its global workforce — around 600 employees.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> PayPal
- Here’s how Rainforest, a budding Stripe rival, aims to win over software companies -MENTIONS-> PayPal
- Discogs’ vibrant vinyl community is shattering -MENTIONS-> PayPal
- Thiel -WORKS_FOR-> PayPal
- Hoffman -WORKS_FOR-> PayPal

Entities: PayPal, Alphabet, tech layoffs, A comprehensive list of 2023 tech layoffs, The Information, Groupon, Impossible Foods, Arrival, Waymo, Blind, Groupon
layoffs, Impossible Foods layoffs, PayPal layoffs, Arrival layoffs, Waymo layoffs
</item 4>

<item 5>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Prior to the layoff, TuSimple had about 550 employees in the U.S. and post reduction in force the company will have about 220, according to the company.

Nuro

Announced May 12 that it will lay off 30%, or about 340 employees across the company.

LinkedIn

Announced May 8 that it is cutting 716 jobs, or about 3.6% of total employees, and will phase out its local jobs app in China. Despite reducing some roles,
LinkedIn also plans to open about 250 new jobs on May 15.

Rapid

Announced May 5 that Rapid (previously known as RapidAPI) has laid off another 70 employees less than two weeks after letting go of 50% of its staff. Just 42
people remain at the company, down from 230 in April, dropping a total of 82% in headcount.

Meesho

Announced May 4 that it has cut 15% of its workforce, or 251 roles.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, LinkedIn, TuSimple, Nuro, local jobs app, Rapid, Meesho, TechCrunch, China, U.S.
</item 5>

<item 6>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Bandcamp

Eliminated 50% of its staff on October 16 after ownership changed from video game company Epic to music licensing platform Songtradr.

LinkedIn

Confirmed October 16 that it will cut 668 more jobs, bringing the total to nearly 1,400 this year, with the bulk of the most recent cuts impacting those with
R&D roles.

Flexport

On October 13 began laying off 20% of its workers, about 600 people, as it prepares for restructuring, and continues a cost-cutting campaign that began with
rescinding offers and working to lease office space.

Qualcomm

Is cutting 1,258 jobs in two of its California offices. According to a filing with the California Employment Development Department, the semiconductor company
plans to make the reductions in December.

Blue Origin

Laid off 40 people October 10, including software engineers and program managers, according to The Information.

Stitch Fix

Will lay off 558 employees after the personal styling subscription service closes a distribution center in Dallas.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Bandcamp, Songtradr, Bandcamp staff layoffs, LinkedIn, LinkedIn job cuts, Flexport layoffs, Qualcomm,
Qualcomm job cuts, California Employment Development Department, Blue Origin, Blue Origin layoffs, The Information, Stitch Fix, Stitch Fix layoffs
</item 6>

<item 7>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Tropic

Eliminated 26 roles, CEO David Campbell confirmed in a LinkedIn post October 21.

Convoy

Is halting operations due to a “massive freight recession,” CEO Dan Lewis announced in a memo October 19, a week after laying off 20% of its workforce.

LegalZoom

Plans to lay off more than 100 workers in its sales division, the company disclosed in a Worker Adjustment and Retraining Notification Act letter.

Plume

Laid off more than two dozen workers, TechCrunch learned October 18, about a year after the trans healthcare startup raised a $24 million Series B.

Google

Eliminated at least 40 roles in its news division, Google confirmed October 18.

Expedia Group

Cut around 100 jobs, Skift reports, in its second round of layoffs in recent months. The cuts reportedly impacted employees across data and AI.

Stack Overflow

Laid off 28% of its staff, more than likely impacting 100 people, the Prosus-owned company announced October 16.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Tropic, David Campbell, LegalZoom, Plume, Stack Overflow, Prosus, Skift, trans healthcare, layoffs,
LinkedIn, TechCrunch, AI, data, Google
</item 7>

<item 8>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
C3.ai

Reportedly cut jobs across departments November 20, citing a need to reduce costs. People familiar with the matter told Bloomberg that several impacted
employees received just one month of severance.

Jodo

Reportedly cut around 100 roles across engineering, customer success, data and sales November 20.

Amazon

Is reportedly laying off “several hundred” employees in their Alexa division and its freshly launched Artificial General Intelligence team.

FreshBooks

Laid off 6% of their workforce November 17 as the company shuts down operations in Raleigh, North Carolina. President Mark Girvan and CEO Don Epperson are also
leaving the company.

Beamery

Is reportedly cutting 25% of its workforce amid a larger restructuring and cost-cutting effort.

Paystack

Laid off 33 employees in Europe and Dubai November 16 as the company cuts down on operations outside of Africa.

Sega

Is reportedly planning to “phase out” 80 temporary workers by 2024.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Chapter 11 bankruptcy
- A comprehensive list of 2023 tech layoffs -MENTIONS-> IronNet
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Turkey
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Yahoo
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Mark Zuckerberg

Entities: A comprehensive list of 2023 tech layoffs, Jodo, Alexa division, Artificial General Intelligence team, FreshBooks, Raleigh, Mark Girvan, Don Epperson,
Beamery, Paystack, Sega, TechCrunch, North Carolina, Europe, Bloomberg
</item 8>

<item 9>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Superpedestrian

Is shutting down its U.S.-based shared scooter operation and exploring a sale of its European business, TechCrunch has exclusively learned.

Bolt

Confirmed it laid off 29% of its staff December 15. The e-commerce company known for its one-click checkout technology was at one time the subject of a federal
probe.

Cruise

In an announcement obtained exclusively by TechCrunch, the embattled self-driving car company is cutting 900 employees, or 24% of its workforce, per a December
14 email from new president and CTO Mo Elshenawy.

Etsy

Is laying off 11% of its workforce. In a letter to employees December 13, CEO Josh Silverman cited a “very challenging macro and competitive environment.”

Chipper Cash

Has reportedly laid off 15 people across various departments. It would be the fintech unicorn’s fourth round of layoffs this year.

SmileDirectClub

Has shut down three months after filing for Chapter 11 bankruptcy, citing $900 million of debt.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Chapter 11 bankruptcy
- Alameda -PARTICIPATED_IN-> Chapter 11 bankruptcy
- FTX -PARTICIPATED_IN-> Chapter 11 bankruptcy
- Chapter 11 bankruptcy -LOCATED_IN-> U.S.
- 2023 in social media: the case for the fediverse -MENTIONS-> Etsy

Entities: Chapter 11 bankruptcy, Superpedestrian, Etsy, SmileDirectClub, Mo Elshenawy, Josh Silverman, federal probe, A comprehensive list of 2023 tech layoffs,
Chipper Cash, Bolt, TechCrunch, Cruise
</item 9>

<item 10>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Chipper Cash relieved almost one-third of its workforce, about 100 employees.

Evernote

On February 17 the company confirmed laying off 129 people.

Jumia

Announced February 16 that it cut 20% of its staff, or more than 900 positions across its 11 markets, in Q4 of 2022.

Convoy

Announced on February 16 that it is shuttering its Atlanta office and laying off workers as part of restructuring. This is the third time in less than a year
that the company has laid off workers.

Sprinklr

Announced on February 15 that it will impact 4% of its global workforce — or more than 100 employees.

iRobot

Announced on February 13 that it will lay off 7% of its workforce, roughly 85 employees.

Twilio

Announced on February 13 that it will impact around 17% of its global workforce, about 1,400 people.

Graph facts:
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> iRobot
- iRobot -MENTIONS-> intense competition
- The 26 best Amazon Prime Day deals you can still shop for today -MENTIONS-> iRobot
- Amazon -SAID-> iRobot
- iRobot -PARTICIPATED_IN-> Amazon’s iRobot purchase

Entities: iRobot, tech layoffs, A comprehensive list of 2023 tech layoffs, Chipper Cash, relieved almost one-third of its workforce, about 100 employees,
Evernote, confirmed laying off 129 people, Jumia, cut 20% of its staff, or more than 900 positions across its 11 markets, in Q4 of 2022, shuttering its Atlanta
office, laying off workers as part of restructuring, Sprinklr, impact 4% of its global workforce, or more than 100 employees, lay off 7% of its workforce,
roughly 85 employees, Twilio
</item 10>
=== END ITEM id=cbc3b621d78a31c1 ===
