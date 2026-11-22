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

This file has 5 items.

=== ITEM id=7e6d431d113d7ecc ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury.
  Fact 0.1: Rehn declared to the jury that all of that, all of it, was built on lies.
Sentence 1: Binance has made headlines this past year for a range of reasons, including Zhao’s comments contributing to the collapse of FTX, which was once one of its top competitors.
  Fact 1.0: Binance has made headlines this past year for a range of reasons.
  Fact 1.1: Binance has made headlines this past year including because of Zhao's comments.
  Fact 1.2: Zhao's comments contributed to the collapse of FTX.
  Fact 1.3: FTX was once one of Binance's top competitors.

Retrieved items:
<item 1>
SBF Trial: The latest updates from the FTX collapse’s courtroom drama (TechCrunch, 2023-10-06T17:22:02+00:00)
The prosecution painted a picture of an empire “built on lies” and made a point to show actual pictures of SBF alongside rich and powerful figures to illustrate
the peak of his power and influence prior to FTX’s implosion.

“This man stole billions of dollars from thousands of people,” prosecutor Thane Rehn told the court. “He defrauded sophisticated investors and lenders, and he
emptied the accounts of ordinary customers, too. He bought himself wealth, power and influence.”

The defense, on the other hand, depicted SBF and his colleagues as in over their heads, without ill-will as FTX collapsed around them.

“Sam and his colleagues were building the plane as they were flying it,” defense attorney Mark Cohen said. “They had to figure out how to navigate a world where
they were running FTX, building out its systems, dealing with hacking threats, managing the credit risk of their customers, managing hundreds of employees, all
while building up their actual exchange.”

Read a recap of the prosecution and defense’s full arguments here.
</item 1>

<item 2>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 3>

<item 4>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 4>

<item 5>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
regulators and government officials, many of whom he donated to.

Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).

But in early November 2022, that all changed.

FTX’s collapse

Concerns surrounding FTX’s liquidity grew after CoinDesk published a copy of Alameda’s balance sheet, showing the firm held $14.6 billion in assets and $8
billion in liabilities as of June 30, 2022.

But there was a problem: The report showed Alameda’s largest asset was $3.66 billion of “unlocked FTT” and $2.16 billion of “FTT collateral.” FTT was the token
behind FTX.

The balance sheet showed that the $5.82 billion in FTT tokens that Alameda owned was 193% higher than the total FTT market cap, which was about $3 billion at
the time.
</item 5>

<item 6>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
"FTX defrauded all its customers," he said.

Matt Huang, co-founder of Paradigm Operations LP, right, arrives at court in New York, US, on Thursday, Oct. 5, 2023. Former FTX Co-Founder Sam Bankman-Fried is
charged with seven counts of fraud and money laundering following the collapse of his cryptocurrency empire last year. Photographer: Yuki Iwamura/Bloomberg via
Getty Images Yuki Iwamura | Bloomberg | Getty Images

Investment to zero

The third witness to take the stand was Matt Huang, co-founder and managing partner of Paradigm, a crypto venture capital firm that invested over $275 million
in FTX. That stake was wiped out. Huang testified about his firm's due diligence on FTX, and he told the court that Bankman-Fried assured him that funds would
be used for FTX and not Alameda.
</item 6>

<item 7>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
That means it purported to have more FTT tokens on its balance sheet than what existed in the world.

Around the same time it was exposed, the world’s largest crypto exchange, Binance, started pulling out its remaining $2.1 billion equivalent of cash in BUSD and
FTT. (It had an equity position in FTX from 2019 to 2021.) This essentially triggered a bank run on FTX.

FTX and Alameda filed for Chapter 11 bankruptcy in the U.S. mid-November 2022. Bankman-Fried resigned, and John J. Ray III, the Enron turnaround veteran, was
appointed its new CEO.

Bankman-Fried, however, maintained his innocence. At The New York Times’ DealBook Summit, he appeared virtually from the Bahamas, saying “I didn’t ever try to
commit fraud on anyone; I was shocked by what happened this month.”
</item 7>

<item 8>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 8>

<item 9>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
It is easy to lie with words and harder to lie with money. Bankman-Fried went around telling people that he wanted to make a lot of money to give it away, an
idea of the Effective Altruism movement that Bankman-Fried affiliated himself with — but the amount of money he funneled to his Paper Bird vehicle alone was ten
times the size of his philanthropic spending as outlined by Easton. And it was all vastly outweighed by his investments — the places where he could prove he was
very smart by investing in the right companies early.

And in brief, explosive testimony, former FTX lawyer Can Sun demonstrated what Bankman-Fried’s tactic was. During the days in November when the exchange
imploded, Bankman-Fried was trying to raise money from Apollo, the private equity group. Apollo wanted an explanation for what had happened, and so
Bankman-Fried came to Sun to ask him to come up with justifications for the missing funds.
</item 9>

<item 10>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Here are the hits:

Bankman-Fried gave interviews freely — and quickly rose to public prominence in the industry. Though FTX hadn’t been in the business as long as competing
exchanges such as Coinbase, Kraken, or Gemini, Bankman-Fried positioned himself as an important, boyish face for crypto. (At one point, Bankman-Fried told a
colleague at FTX that “I honestly think it’s negative EV [this may mean “expected value,” as in poker] for me to cut my hair. I think it’s important for people
to think I look crazy.”)

Because he was so successful at this kind of public relations, his fall from grace was another mark against an industry that was already roiled by bankruptcies
and scandals. Some additional trouble for the crypto industry is likely to come from one crucial element of the fraud trial — the part where the government must
prove intent.

The first part of proving the government’s case is pretty simple and a little boring: prosecutors must show that certain transactions took place.
</item 10>
=== END ITEM id=7e6d431d113d7ecc ===

=== ITEM id=a4b8e82e31f44964 ===
Gold sentences and facts:
Sentence 0: The feature is currently available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.
  Fact 0.0: The feature is currently available in France.
  Fact 0.1: The feature is currently available in Germany.
  Fact 0.2: The feature is currently available in Japan.
  Fact 0.3: The feature is currently available in the U.K.
  Fact 0.4: The feature is currently available in the U.S.
  Fact 0.5: There are plans for further expansion of the feature.
Sentence 1: LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
  Fact 1.0: LinkedIn leans into AI tools.
  Fact 1.1: LinkedIn this week unveiled a string of new AI features.
  Fact 1.2: The new LinkedIn AI features span its job hunting products.
  Fact 1.3: The new LinkedIn AI features span its marketing products.
  Fact 1.4: The new LinkedIn AI features span its sales products.
  Fact 1.5: Ingrid writes about LinkedIn's new AI features.

Retrieved items:
<item 1>
6 VCs explain how startups can capture and defend marketshare in the AI era (TechCrunch, 2023-10-13T20:01:40+00:00)
You cannot escape conversations about AI no matter how far or fast you run. Hyperbole abounds around what current AI tech will be able to do (revolutionize
every industry!) and what current AI tech will be able to do (take over the world!). Closer to the ground, TechCrunch+ is working to understand where startups
might find footholds in the market by levering large language models (LLMs), a recent and impactful new method of creating artificially intelligent software.

How AI will play in startup land is not a new topic of conversation. A few years back, one venture firm asked how AI-focused startups would monetize and whether
they would suffer from impaired margins due to costs relating to running models on behalf of customers. That conversation died down, only to come roaring back
in recent quarters as it became clear that while LLM technology is quickly advancing, it’s hardly cheap to run in its present form.

But costs are only one area where we have unanswered questions.
</item 1>

<item 2>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
The idea, O’Hara said, is to rethink social networking from the ground up.

Ford shutters SaaS app for field work: Ford has shut down VIIZR, a software-as-a-service company that, along with Salesforce, built an app to help tradespeople
like plumbers, locksmiths and electricians to schedule field appointments, send invoices and manage customers, Kirsten reports. VIIZR, which was announced in
December 2021, was a separate company majority owned by Ford, with Salesforce as a minority investor.

Apple bets on generative AI: Apple CEO Tim Cook pushed back against the notion that the company was behind in AI on Apple’s Q4 earnings call with investors, as
he highlighted technology developments that Apple had made recently that “would not be possible without AI.” Cook also said that Apple was working on generative
AI technologies, lending credence to reports suggesting the company is on track to spend $1 billion per year on developing generative AI products.
</item 2>

<item 3>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
It is aiming to help any software play nice with ChatGPT and other LLM ecosystems.

What big eyes you have: Adobe continues to push for easier image editing, showing off its Project Stardust as a sneak preview of its next-gen AI photo editing
engine.

Holding back the tide: Creatives across industries are strategizing in a campaign that targets potential corporate abuse of AI technology. The conglomeration is
realistic about the ways that musicians and some other creatives could benefit on an individual level from automating parts of their work. The goal is that AI
tools “become ways for individual humans to make more money, work less, and compete with the corporations that exploit them.”

Tech you can touch

A while back I argued that Apple’s new AR headset is a game-changer for startups. It seems that’s likely the case at the high end. But more in the realm of
affordability, Meta Quest 3 takes a step closer to mainstream AR/VR, Brian reports.
</item 3>

<item 4>
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
</item 4>

<item 5>
8 predictions for AI in 2024 (TechCrunch, 2023-12-19T16:19:34+00:00)
The “app store for AI” will be pushed hard as the platform to get your AI toys and tools from, and never mind Hugging Face or any open source models. They have
an excellent model to work from, Apple’s, and will follow it all the way to the bank.

Expect more moves like that from 2024’s OpenAI as the caution and academic reserve that the previous board exerted gives way to an unseemly lust for markets and
customers.

Other major companies with AI efforts will also follow this trend (for instance, expect Gemini/Bard to horn in on a ton of Google products), but I suspect it
will be more pronounced in this case.

Agents, generated video and generated music graduate from quaint to experimental

Some niche applications of AI models will grow beyond “eh” status in 2024, including agent-based models and generative multimedia.

If AI is going to help you do more than summarize or make lists of things, it’ll need access to things like your spreadsheets, ticket buying interfaces,
transportation apps and so on.
</item 5>

<item 6>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 6>

<item 7>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 7>

<item 8>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 8>

<item 9>
If your startup doesn’t seem impossible at first, it’s not hard enough (TechCrunch, 2023-11-03T19:05:30+00:00)
The latest funding round brings QI Tech’s total capital raised to $262 million.

New Web Summit CEO appointed: Former Wikimedia Foundation CEO Katherine Maher has been appointed as the new CEO of Web Summit, following the controversial
departure of its previous leader, Paddy Cosgrave. Maher’s appointment comes as the tech conference faces a challenge to rebuild relationships and trust within
the tech community.

Top reads on TechCrunch this week

Shinier, smarter maps: Google is rolling out a variety of updates for Maps, incorporating AI technology to enhance user experience. The feature is currently
available in France, Germany, Japan, the U.K. and the U.S., with plans for further expansion.

From stop-and-start to charging ahead: Over on TC+, Tim has had it with U.S. automakers hesitating on EVs, as Ford and GM are showing signs of backing away from
their commitment to electric vehicles. Cowards.
</item 9>

<item 10>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Hey, folks, welcome to Week in Review (WiR), TechCrunch’s regular newsletter that recaps the past few days in tech. AI stole the headlines once again, with tech
giants from Google to X (formerly Twitter) heading off against OpenAI for chatbot supremacy. But plenty happened besides.

In this edition of WiR, we cover Google faking a demo of its new AI model (and giving out offensive notebooks to Black summit attendees), defense startup
Anduril unveiling a fighter jet weapon, the continued fallout from the 23andMe hack, and the release of the Grand Theft Auto VI trailer. Also on the roster are
stories about patient scans and health records spilling online, Meta’s new AI-powered image generator, Spotify cutting jobs and an autonomous truck startup
leaving the U.S.

It’s a lot to get to, so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t already done so.
</item 10>
=== END ITEM id=a4b8e82e31f44964 ===

=== ITEM id=609812b5ee0951c5 ===
Gold sentences and facts:
Sentence 0: The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive thinking.
  Fact 0.0: The pastor explained that Jesus brought more positivity than anyone.
  Fact 0.1: The pastor said it is important to keep faith aligned with the practice of positive thinking.
Sentence 1: He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically.
  Fact 1.0: The pastor added that when a person consistently taps into the habit of positive thinking, their relationship with Jesus grows deeper and deeper.
  Fact 1.1: The pastor said the growth in that relationship happens automatically.

Retrieved items:
<item 1>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
A Harvard study shares that optimism reduces the risk of dying prematurely — from overcoming trauma, making the most of your opportunities to overall life
satisfaction, positivity makes a difference."

WASHINGTON PASTOR ACKNOWLEDGES THE PAIN OF PUTDOWNS AGAINST THE CHRISTIAN FAITH: ‘ASK GOD FOR COURAGE’

The pastor said he wrote his most recent book, "The Power of the Second Thought," to help people gain habits to strengthen their mental health.

Bradley graduated from Dartmouth College and was playing professional soccer when he became ill — something that took 10 years to fully recover from. He said
finding hope during that time was challenging but is how he changes his "hopeless first thoughts."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain," he said.

FAITH DILEMMA IN FOCUS: ‘IT’S SUNDAY AND I HAVEN'T GONE TO CHURCH — WHAT DO I DO?'
</item 1>

<item 2>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Bradley said, "Positive thinking is relational, and there is no greater source of joy in our lives than God."
</item 2>

<item 3>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
FIRST SUNDAY OF ADVENT IS A REMINDER TO ‘WATCH’ AND PREPARE FOR CHRIST'S SECOND COMING, SAYS FAITH LEADER

There are thousands of prayers on this app. And I'm instantly transported into a sacred space, no matter where I am.

No matter what, if you live in a major city or if you live in a place that's just really busy, you can pop on your headphones and connect to the app. You start
praying with a meditation and you're instantly just transported.

This is something that I've been so grateful for — and especially as we go into the hubbub and the busyness of the Christmas season, it's something that I'm
relying on daily, to be able to just cut through all that noise.

‘Transformative for people’

Question: Are people reaching out to you about your involvement with the app and asking you more about it?
</item 3>

<item 4>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Reject and replace!"

He gave this list of examples:

Replace "People just slack and gossip at my job" with "I will work with all my strength, even when no one is looking"

Replace "There's no hope for my future" with "I am thankful for today and will keep doing the next right thing"

Replace "My children are annoying" with "My children are a gift from God"

Replace "I'll never have good friendships" with "I will pursue and support the people that I respect the most"

Replace "No one really cares about me" with "God is with me and loves me every day"

Said Bradley, "Johns Hopkins reports that there are links between positive thinking and health. People with a positive outlook are less likely to have a heart
attack or cardiovascular event."

"I didn't have a roadmap for finding hope or the tools to handle the intensity of my loss and pain."

Also, "Research from the University of North Carolina points to positive people being more resilient and finding more solutions to problems.
</item 4>

<item 5>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
We do not have to believe, harbor or entertain negative first thoughts. We can be proactive by intentionally choosing our second thought."

It is no accident that the pastor's new book, his second, is called "The Power of the Second Thought."

It dives into the power of understanding that the first thoughts we have are not always the best — and that it's wise to take some time and get to the "second
thoughts," which can yield greater understanding.

STORYTELLER REVEALS HIS FAITH AT WORK: ‘I REFUSE TO TRY TO SCREW SOMEONE IN THE NAME OF BUSINESS’

Said Bradley to Fox News Digital in emailed comments this week, "How can we choose a positive attitude and approach during the day? Reject and replace!"
</item 5>

<item 6>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
He said turning to Jesus for a deeper meaning and learning to not take his "first thoughts" for fact was crucial to his overall mental health.

"The Word of God is living and active, bringing strength, courage, wisdom, and protection," he said.

"God's thoughts and ways are higher than ours, but when we spend time, each day reading His Word and listening to the voice of the Good Shepherd, we are filled
with new thoughts that change the trajectory of our lives."

Bradley also shared his dos and don'ts of positive thinking.

He said it's not "denial and pretending that there are no challenges," he said.

"Positive thinking is not flattery and frivolous compliments."

CLICK HERE TO SIGN UP FOR OUR LIFESTYLE NEWSLETTER

Rather, positive thinking is realistic, full of truth and love — and aligned with heaven and scripture.

"Positive thinking is not self-consumption, self-absorption or selfishness. Positive thinking is not creating unrealistic expectations for other people and
putting extra stress on them to be perfect."
</item 6>

<item 7>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
HALLOW APP ANNOUNCES COLLABORATION WITH LIAM NEESON FOR NEW ADVENT SERIES THIS YEAR

People can "disconnect from the chaos of the real world, the divisiveness, the anxiety" that exists, said Roumie, by connecting with Jesus through prayer.

They can find "peace and a deeper relationship with Christ, and I think that's really the goal of the prayer challenge," he said.

Roumie said that building a habit of prayer "is easier than people think."

He added, "As you consistently tap into that habit, you find your relationship with Jesus growing deeper and deeper, just automatically. And as a result, you
then experience much more peace, tranquility and serenity in your life and the ability to really muscle through some of the most difficult obstacles and
challenges that life can throw at you," he said.
</item 7>

<item 8>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Positive thinking is not creating unrealistic expectations for other people and putting extra stress on them to be perfect."

Bradley said, "Positive thinking is a vital aspect of loving your neighbor, nourishing your mind, discerning between light and darkness, good and evil and right
and wrong."

"Positive thinking leads to taking action and improving situations. Positive thinking brings out the best in the people around you. Positive thinking rejects
excuses, takes responsibility, and overcomes setbacks."

The pastor went on to explain that Jesus brought more positivity than anyone — which is why it's important to keep faith aligned with that practice of positive
thinking.

CLICK HERE TO GET THE FOX NEWS APP

"Jesus exudes a trustworthy positivity that is full of meaning, faithfulness, and transformation," he noted.

"God renews our thinking and vision, empowers us to make intentional choices about our focus and perspective and trains us how to win the battle of our minds,"
he added.
</item 8>

<item 9>
Faith dilemma: How effective is positive thinking when life gets difficult? Pastor weighs in (FOX News - Lifestyle, 2023-10-28T09:00:12+00:00)
Pastor Jesse Bradley of Auburn, Washington runs Grace Community Church.

He's ministered to many, many people struggling with a variety of faith dilemmas and issues in their lives.

In this month of October 2023, as Americans grapple with a variety of issues, he addressed the notion of positive thinking — and shared with Fox News Digital
what it would "look like to change your thinking and shift in a new direction."

WASHINGTON PASTOR, ONCE GRAVELY ILL, SHARES HOW ‘THE POWER OF THE SECOND THOUGHT’ TRANSFORMED HIS LIFE

Said Pastor Bradley, "Psychologists report that we have in between 6,000 and 60,000 thoughts a day, based on the person and situation. The National Science
Foundation shares that up to 80% of our first thoughts are not helpful, even destructive."

What happens is that "when life is difficult, we can become flooded with thoughts that are drenched in fear, worry, stress, selfishness, impurity, selfishness
and lies.
</item 9>

<item 10>
Jonathan Roumie on faith and prayer life during Advent and always: 'God hasn't forsaken us' (FOX News - Lifestyle, 2023-12-05T09:00:54+00:00)
CLICK HERE TO GET THE FOX NEWS APP

So it's challenged me on that level.

But as it challenges me, it also allows me to go further in that relationship — and to deepen my prayer life.
</item 10>
=== END ITEM id=609812b5ee0951c5 ===

=== ITEM id=adb35897a9d28a8b ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: In blog posts and press materials, Google touted Gemini's superior architecture.
  Fact 0.1: In blog posts and press materials, Google touted Gemini's superior capabilities.
  Fact 0.2: Google claimed that Gemini meets or exceeds the performance of other leading gen AI models.
  Fact 0.3: OpenAI's GPT-4 is one of the other leading gen AI models Google compared Gemini to.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: This point will probably come up later in the trial.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.
</item 1>

<item 2>
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
</item 2>

<item 3>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
Epic Games will specifically reference one agreement it had made with the OEM OnePlus, to make Epic games available on its devices through the Epic Games app.
But OnePlus had to cancel the deal over concerns that the arrangement would have it bypassing the Google Play Store. The case will also examine a deal between
Google and Samsung that Epic says was designed to prevent the Galaxy Store from being a competitive threat and require Google Play Billing on apps distributed
by Samsung.

Epic will argue Google’s developer agreements are anti-competitive

A second set of agreements Epic will highlight are those between Google and Android app developers themselves. One such program, the “Project Hug” initiative, a
part of the Google Games Velocity Program, will be cited as an example of where Google paid competitors not to distribute their apps on Android outside of the
Play Store by targeting those that were most at risk of attrition from Play, then approaching them with a monetary offer.
</item 3>

<item 4>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 4>

<item 5>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 5>

<item 6>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
This week, Google took the wraps off of Gemini, its new flagship generative AI model meant to power a range of products and services including Bard, Google’s
ChatGPT competitor. In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds
the performance of other leading gen AI models like OpenAI’s GPT-4.

But the anecdotal evidence suggests otherwise.

A “lite” version of Gemini, Gemini Pro, began rolling out to Bard yesterday, and it didn’t take long before users began voicing their frustrations with it on X
(formerly Twitter).

The model fails to get basic facts right, like 2023 Oscar winners:

I'm extremely disappointed with Gemini Pro on Bard. It still give very, very bad results to questions that shouldn't be hard anymore with RAG. A simple question
like this with a simple answer like this, and it still got it WRONG.
</item 6>

<item 7>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 7>

<item 8>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 8>

<item 9>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 10>
=== END ITEM id=adb35897a9d28a8b ===

=== ITEM id=59bc21d83aa30bb6 ===
Gold sentences and facts:
Sentence 0: Today, social networks are often run by large corporations — like Meta, Snap and Google — where advertisers pay the bills.
  Fact 0.0: Today, social networks are often run by large corporations.
  Fact 0.1: Meta, Snap and Google are examples of large corporations that run social networks.
  Fact 0.2: On social networks run by large corporations, advertisers pay the bills.
Sentence 1: By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter.
  Fact 1.0: By the end of 2012, Google and Facebook were angling to acquire the London lab.
  Fact 1.1: Three people familiar with the matter said Google and Facebook were angling to acquire the London lab by the end of 2012.

Retrieved items:
<item 1>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
And I think also then access to content that spurs those conversations and seeds those conversations. And it all being kind of a delightful user experience —
but we’re at the early days of that,” she adds.

Another avenue of thought is to create spaces within the fediverse where consumers can have civil and direct conversations within smaller communities, instead
of shouting into the void, so to speak. As to what that will look like, is less clear. Will it be different instances or just new ways of forming communities,
by building out features or experiences? O’Hara couldn’t say, only noting that experimentation will be needed.

Importantly, trust and safety will also be key to Mozilla’s instance. Its content policies indicate strict measures around hate speech, impersonation,
self-harm, harassment, misinformation, violent and sexualized content, and more, in addition to illegal content, like CSAM (child sexual abuse material) and
promotion of illegal goods.

But there’s a chance that playing it too safe can make a product less compelling as a Twitter/X alternative.
</item 1>

<item 2>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
He found the iMac to be lacking and not necessarily worth the upgrade from the 2021 model, excepting the M3 chip, which brings “impressive” performance gains
over the already-powerful M1. As for the M3 Max MacBook Pro, Brian reports that, at $2,500 (plus some pricey add-ons), it successfully splits the difference
between the Mac Studio and MacBook Air.

Mozilla bets on a decentralized future: Sarah spoke with Mozilla senior director of content Carolyn O’Hara, who outlined Mozilla’s strategy where it concerns
the “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub protocol.
The idea, O’Hara said, is to rethink social networking from the ground up.
</item 2>

<item 3>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
“[They] aren’t just putting forward the kind of standards that are good for people, but are just good for their bottom lines.”

Plus, she adds, consumers are now becoming aware of this, which is new.

“Consumers feel that the vibes are off a little bit…these platforms aren’t necessarily working in their best interests, or satisfying them,” O’Hara points out.

For Mozilla, that discontent presented an opportunity to enter the social networking space and compete for consumers’ attention. In its case, however, the
broader goal is to help the fediverse itself gain traction, not just its own Mastodon server.

“We’re not seeking to get, like 2 billion users on our instance. We want people to have choice and agency,” says O’Hara, adding that choice is in line with
Mozilla’s values.

However, the company is aiming to tackle some of the obstacles that have prevented users from joining and participating in the fediverse so far, including the
technical hurdles around onboarding, finding people to follow and discovering interesting content to discuss.
</item 3>

<item 4>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
As it learns from the publishers and other content creators what they want and need, it then intends to use that understanding to build features and drive
efforts that can help them reach their audiences.

So far, what Mozilla has learned is that publishers today feel somewhat resigned and pessimistic about social networks. Twitter isn’t driving much traffic and,
even though Instagram Threads appears interesting, Meta specifically said it wasn’t going to prioritize news on that platform. What Mozilla wants to accomplish,
then, is to help reconfigure the Mastodon onboarding process so that when someone — including a publisher or creator — joins its instance (or the fediverse in
general) they’re able to build their audience with more ease.

“The baseline is just better onboarding, better connecting with accounts and communities,” explains O’Hara. “You might be interested in spaces where you can
have the kinds of conversations that you want to be able to have; trust and safety practices that make you feel safe in order to participate. And I think also
then access to content that spurs those conversations and seeds those conversations.
</item 4>

<item 5>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Consumers are hungry for a new way of social networking, where trust and safety are paramount and power isn’t centralized with a Big Tech CEO in charge… or at
least that’s what Mozilla believes. The mission-driven tech company behind the Firefox browser, Pocket reader and other apps is now investing its energy into
the so-called “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub
protocol.

The idea is to rethink social networking from the ground up.

Today, social networks are often run by large corporations — like Meta, Snap and Google — where advertisers pay the bills. This has created a world where modern
social networks are profit-driven, and consumer needs aren’t always at the forefront, Mozilla believes. This disconnect between what people want and what
today’s networks offer hit an inflection point last year, when billionaire Elon Musk bought Twitter, leading to a surge of interest in alternatives to
centralized social networking, including apps like Mastodon and Bluesky.
</item 5>

<item 6>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
That’s one takeaway, at least, from the recent shutdown of Pebble (formerly T2), a Twitter clone that put trust and safety at the forefront of its development
process. The app never gained more than 20,000 users, and the founders now believe that its approach to moderation may have been correct, but it wasn’t a growth
driver. In other words, people may say they want to feel safer when participating online, but clearly, that isn’t their only need.

Mozilla’s instance, still in private testing, has a small team of moderators that will grow when the instance launches to the public in the coming months.

“It’s not just a commitment, it’s beyond foundational for us — that is the thing that is going to be true about this instance,” O’Hara says of Mozilla’s plans
to heavily moderate its space. Though Mozilla will also rely on technology to do some of the work, it wants to make sure the human moderators, “feel protected,
supported and respected for their work,” she adds.
</item 6>

<item 7>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
Under Musk, Twitter — now called X — has prioritized “free speech” managed by crowdsourced-based moderation. Its policies have resulted in increased hate speech
and toxic content on the platform, some studies have shown.

Mozilla believes there’s a better way forward and, more importantly, that now is exactly the right time to invest in that path. And, as a wholly owned
subsidiary of a nonprofit, the company says it’s not motivated by generating earnings for shareholders or returning a VC investment, allowing it to progress
with a collaborative approach where it takes in input from a lot of different voices.

In an interview with TechCrunch, Mozilla Senior Director of Content Carolyn O’Hara explained why Mozilla has taken an interest in the fediverse and Mastodon,
specifically, and how it expects to experiment in the months ahead.
</item 7>

<item 8>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
For background, the company announced its plans to enter the fediverse in December 2022 and this May launched a private beta for Mozilla.social, a Mastodon
server — or “instance” in fediverse parlance — that will allow consumers to participate in this new form of social networking where problems like
misinformation, harassment and issues around worsening mental health are minimized.

The company’s work on its social efforts is also available publicly on GitHub. (Of interest to technical folks, the company is using a forked version of Elk as
its alternative web client for its Mastodon server, which is what gives it its slick look and feel.)

Mozilla’s involvement, says O’Hara, came about because the company looked at the history of social media over the past decade and didn’t like what it saw.

“I think that it’s a pretty poor track record by existing companies that are only model motivated by profit and just insane user growth, and are willing to
tolerate and amplify really toxic content because it looks like engagement,” she says.
</item 8>

<item 9>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter. Hassabis and his co-founders
insisted on two conditions: No DeepMind technology could be used for military purposes, and its AGI technology must be overseen by an independent board of
technologists and ethicists.

Google offered $US650 million. Zuckerberg of Facebook offered a bigger payout to DeepMind’s founders but would not agree to the conditions. DeepMind sold to
Google. Zuckerberg was determined to build an AI lab of his own. He hired Yann LeCun, a French computer scientist who had also done pioneering AI research, to
run it. A year after Hinton’s auction, Zuckerberg and LeCun flew to Lake Tahoe for the same AI conference. While padding around a suite at the Harrah’s casino
in his socks, Zuckerberg personally interviewed top researchers, who were soon offered millions of dollars in salary and stock. AI was once laughed off.
</item 9>

<item 10>
Why Mozilla is betting on a decentralized social networking future (TechCrunch, 2023-11-03T17:43:00+00:00)
“We are saying out the gate that this isn’t a neutral platform,” O’Hara stresses. “We think that that’s often used as this crutch to allow, or even amplify,
really toxic content in the name of engagement… In some cases, platforms aren’t just brave enough to actually just take things down.”

Building directly on the fediverse isn’t the only way the company is prompting new forms of social networking. The company has also financially backed a
startup, Mammoth, the makers of a third-party app for Mastodon.

Mozilla has been leaning into its values around inclusion, dignity, security, agency and community for 25 years, O’Hara says, and those will guide its efforts
in the fediverse, as well.

“We’re going to have rules on how people can engage with one another…it’s the fediverse, you have lots of other places to go if you don’t want to abide by those
rules,” she adds.

Users can provide Mozilla with feedback via @Social @Mozilla.social or by using the hashtag #mozillasocial within the product.
</item 10>
=== END ITEM id=59bc21d83aa30bb6 ===
