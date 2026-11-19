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

=== ITEM id=f2dde049e2c1ceda ===
Gold sentences and facts:
Sentence 0: The ride-hail company claims that the rate of sexual assault reported on the app decreased 38% between its first and second reports.
  Fact 0.0: The ride-hail company claims the rate of sexual assault reported on the app decreased 38%.
  Fact 0.1: The 38% decrease is between its first and second reports.
Sentence 1: The survivors in the joined lawsuit also allege that Uber’s “fast and shallow background checks” are substandard and designed to make it as easy as possible for drivers to sign up quickly.
  Fact 1.0: Survivors in the joined lawsuit allege that Uber's 'fast and shallow background checks' are substandard.
  Fact 1.1: Survivors allege the background checks are designed to make it as easy as possible for drivers to sign up quickly.
Sentence 2: Uber has attempted to address sexual assaults by drivers — which the lawsuits claim Uber has known about since 2014 — through new safety features in its app, like a 911 button and the ability to share location with a friend.
  Fact 2.0: Uber has attempted to address sexual assaults by drivers through new safety features in its app.
  Fact 2.1: The lawsuits claim Uber has known about sexual assaults by drivers since 2014.
  Fact 2.2: One of Uber's new safety features is a 911 button.
  Fact 2.3: Another of Uber's new safety features is the ability to share location with a friend.
Sentence 3: In the third quarter, Uber generated $394 million in operating income and $219 million worth of net income.
  Fact 3.0: In the third quarter, Uber generated $394 million in operating income.
  Fact 3.1: In the third quarter, Uber generated $219 million worth of net income.

Retrieved items:
<item 1>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
The number of incidents increased in the categories of nonconsensual kissing, nonconsensual touching and nonconsensual penetration, or rape, between 2017 and
2019.

The total number of rape incidents also rose from 2017 to 2019. And while the number of reported rapes dropped in 2020, the rate has remained the same since
2017. In other words, there has not been an improvement in the rate of rapes on the platform.

Sexual assaults are an ongoing problem for Uber

Uber has been sued numerous times over the past few years by passengers who claim they were sexually assaulted during a ride. Lyft has faced similar lawsuits
and accusations.

Under Uber’s terms of use, class action lawsuits can’t be filed against the company in cases of sexual assault, so each case has to be heard individually. This
has prevented survivors from advocating for themselves collectively.

Judge Charles Breyer in the Northern District of California will preside over pretrial hearings.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Uber sexual assault survivors call for in-car cameras, tech upgrades, Lyft, Charles Breyer, pretrial hearings, sexual assault, class
action lawsuits, Northern District of California, rape
</item 1>

<item 2>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
After reporting, shares of Uber are up 1.6% in trading, after an up-and-down morning of shares exchange.

Against a backdrop of macroeconomic uncertainty, the former startup darling’s results can be viewed as an indication that its business model has matured and is
now in a stable, profitable and cash-generating position. On the other hand, under-performing demand results from a company as global as Uber could indicate
that consumer spend is coming in lighter than anticipated.

For startups in the transport and on-demand sectors, Uber’s earnings are a regular informational zeitgeist. So let’s work to understand where Uber’s revenue
came from in the third quarter, and how each of its top-line sources converted — or not — to bottom-line results.

Where did the money come from?

In the third quarter, Uber saw total bookings rise from $29.1 billion to $35.3 billion, a gain of around 21%.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, transport and on-demand sectors, macroeconomic uncertainty, consumer spend, Uber’s Q3 numbers include impressive profitability gains,
slower-than-expected growth, Uber
</item 2>

<item 3>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Hundreds of women have filed lawsuits against Uber claiming the company hasn’t done enough to prevent instances of sexual assault by drivers. Now a panel of
judges has ruled that more than 80 cases can be consolidated into federal court.

The stakes are high for both parties with implications extending to future Uber riders and drivers. The outcome of the case could result in sweeping changes to
Uber’s platform, which plaintiffs argue will reduce sexual assault and also raise new concerns over privacy.

The upshot? Technology lies at the center of this very human story.

Uber has attempted to address sexual assaults by drivers — which the lawsuits claim Uber has known about since 2014 — through new safety features in its app,
like a 911 button and the ability to share location with a friend. Yet survivors, and their attorneys, say that response has been inadequate, and they’re
calling for better technological solutions like in-vehicle surveillance cameras.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Federal Court, technology, Uber’s platform, Uber’s app, lawsuits against Uber, 911 button, location sharing with a friend, in-vehicle
surveillance cameras, failure to prevent sexual assault, Uber, Uber sexual assault survivors call for in-car cameras, tech upgrades, sexual assault, privacy
</item 3>

<item 4>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
In business segment terms, Uber generated $17.9 billion in ride-hailing bookings (+31% year over year), and $16.1 billion worth of delivery bookings (+18%).
Those key business groups at the company generated $5.1 billion and $2.9 billion in revenue, respectively, during the September quarter.

There’s nuance to the revenue figures that we need to consider as the rule-changes in question do affect startups that operate in related categories. On the
ride-hailing front, Uber told its investors that its Q3 2023 revenue was “negatively impacted by business model changes in some countries that classified
certain sales and marketing costs as contra revenue by $161 million.” That was not the only legal change that led to Uber’s results changing shape. Under its
delivery business results, the company added that its revenue result was “negatively impacted by business model changes that classified certain sales and
marketing costs as contra revenue by $360 million.”

The combined impact of those two items was 8 percentage points of growth.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, ride-hailing, delivery, business model changes,
contra revenue, Uber, startups
</item 4>

<item 5>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Aside from the 911 button and location-sharing feature, which were both introduced in 2018, Uber has added other features to the app over the last five years.
In 2021, Uber introduced a feature that allows riders and drivers to record audio during a trip. The following year, Uber launched a pilot to provide passengers
with live help from an ADT safety agent, as well as PIN verifications to ensure passengers are connected to the correct driver.

Rachel Abrams — a sexual assault attorney and partner at law firm Peiffer Wolf Carr Kane Conway & Wise, one of the law firms working on the multidistrict
litigation, who filed the petition to consolidate the various actions in federal court — argues the app-based solutions are half-measures that haven’t quelled
instances of sexual violence against passengers on the platform. Mandatory in-vehicle cameras in Uber cars is “essential for safety,” she says.

Uber told TechCrunch it cannot comment on pending litigation but said it remains committed to the safety of all users on its platform.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, sexual violence, Uber, Rachel Abrams, Peiffer Wolf Carr Kane Conway & Wise, Uber app, 911 button, location-sharing feature, audio
recording feature, pilot for live help from an ADT safety agent, ADT, PIN verifications, multidistrict litigation, user safety, mandatory in-vehicle cameras
</item 5>

<item 6>
Graph query row:
article: California’s privacy watchdog eyes AI rules with opt-out and access rights
published: 2023-11-27T17:30:24+00:00
publisher: TechCrunch
related_entity: safety
</item 6>

<item 7>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Aside from in-vehicle surveillance and more extensive background checks, the survivors are asking Uber to implement driver training on interactions with
passengers, a zero-tolerance policy for drivers, sexual harassment education and training, and a more adequate system to encourage customer reporting and
monitor customer complaints.

Are Uber’s in-app safety features working?

Abrams argues that Uber’s safety features are insufficient because they’re all app-based and only potentially useful if a rider has access to their own phone.

“The majority of clients I represent and the women I’ve spoken to who have been sexually assaulted have experienced: (A) someone else ordered the ride for them;
(B) their phone is dead or they can’t locate their phone to use the safety shield; or (C) if they could use their phone, they’re incapacitated,” said Abrams.

The lawyer says she has interviewed upward of 5,000 survivors over the course of several years and has seen no difference in the number of attacks before or
after implementation of these in-app features.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber in-app safety features, sexual assault
</item 7>

<item 8>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
Uber hasn’t released safety data for 2021 and 2022, and won’t say when it plans to do so. Older data backs up some of Abrams’ claims.

Uber has published two safety reports — one containing data from 2017 to 2018 and another with data from 2019 to 2020. The ride-hail company claims that the
rate of sexual assault reported on the app decreased 38% between its first and second reports.

While the rate decrease is positive, not all the data delivers the same message. Over that same time period, the total number of sexual assault reports across
five categories fell from 5,981 to 3,824. That drop could be explained by the decrease of trips from 1.4 billion in 2019 to 650 million in 2020.

And while the total number of incidents as a percentage of total number of rides does decrease through the years, the number of incidents in certain categories
actually increases.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Abrams, Uber sexual assault survivors call for in-car cameras, tech upgrades, Uber safety reports, sexual assault
</item 8>

<item 9>
Uber sexual assault survivors call for in-car cameras, tech upgrades (TechCrunch, 2023-10-13T19:26:48+00:00)
And so if they don’t have drivers, they don’t make money.”

Many drivers install their own dash cams to record trips, usually as backup evidence for insurance claims or to defend themselves against unfair deactivations
from Uber’s platform. Uber is also piloting a new video recording feature for drivers that allows them to record video on their smartphones. But in those cases,
the driver can decide what and when to record, and when to share that data.

Other demands

The survivors in the joined lawsuit also allege that Uber’s “fast and shallow background checks” are substandard and designed to make it as easy as possible for
drivers to sign up quickly. Uber uses third-party companies like Checkr and Appriss to do background checks, which Sergio Avedian, senior contributor at The
Rideshare Guy, says “are at best watered down and not guaranteed of bad apples from falling through the cracks.”

The lawsuit calls on Uber to also include fingerprinting, which would run prospective drivers through FBI databases.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, Uber, Checkr, Appriss, Sergio Avedian, The Rideshare Guy, FBI, background checks, video recording feature, joined lawsuit, Uber sexual
assault survivors call for in-car cameras, tech upgrades, fingerprinting, sexual assault
</item 9>

<item 10>
Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth (TechCrunch, 2023-11-07T17:03:41+00:00)
Uber reported third-quarter earnings on Tuesday that show a profitable ride-hail and delivery company that’s chugging along in spite of slowing growth in some
sectors.

The company reported revenue of $9.3 billion, an 11% increase year over year. Investors had expected Uber to report revenues of around $9.5 billion (FactSet,
Refinitiv), meaning that despite the company’s growth, it fell short of estimates. Turning to profitability, Uber reported net income of $221 million in the
third quarter, or 10 cents per share, compared with a net loss of $1.2 billion, or 61 cents per share, in the same quarter last year. Again, the company fell
short of expectations that it would generate 12 cents in per-share profit.

Looking ahead, Uber anticipates gross bookings of $36.5 billion to $37.5 billion, up just 6% at the top-end from its Q3 result.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -PUBLISHED_BY-> TechCrunch
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -PUBLISHED_BY-> TechCrunch
- More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo -PUBLISHED_BY-> TechCrunch
- After warning, the EU opens investigation into X over alleged disinformation -PUBLISHED_BY-> TechCrunch
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -PUBLISHED_BY-> TechCrunch

Entities: TechCrunch, FactSet, Refinitiv, Uber’s Q3 numbers include impressive profitability gains, slower-than-expected growth, Uber
</item 10>
=== END ITEM id=f2dde049e2c1ceda ===

=== ITEM id=a570a5579ef2ed5d ===
Gold sentences and facts:
Sentence 0: “Certainly there wasn’t a valid alternative to Google at the time,” Cue said.
  Fact 0.0: Cue said there was certainly not a valid alternative to Google at the time.
Sentence 1: Google compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.
  Fact 1.0: Google compared the performance of Gemini Pro to GPT-3.5.
  Fact 1.1: GPT-3.5 is the predecessor of GPT-4.
  Fact 1.2: GPT-3.5 is a model that is around a year old.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> Alphabet
- Earnings week ahead: Amazon, Microsoft, Alphabet, Meta, Ford, GM and more -MENTIONS-> Alphabet
- ASX set to rise ahead of inflation report after big tech lifts Wall Street -MENTIONS-> Alphabet
- Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation -MENTIONS-> Alphabet
- Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas -MENTIONS-> Alphabet

Entities: Alphabet, Helena World Chronicle, U.S. District Court in D.C., Search Generative Experience, Bard AI chatbot, Knowledge Graph, anticompetitive
behavior, U.S. antitrust law, Sherman Act, Arkansas, question-and-answer technologies, News publisher files class action antitrust suit against Google, citing
AI’s harms to their bottom line, class action lawsuit, TechCrunch, AI technologies
</item 1>

<item 2>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, TechCrunch, Early impressions of Google’s Gemini aren’t great, Bard, Edison Ade, N KIRAN KUMAR, Gemini Pro, Google, Gemini
</item 2>

<item 3>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).

Graph facts:
- A comprehensive list of 2023 tech layoffs -MENTIONS-> GitHub
- GitHub -SAID-> streamlining effort
- Jason Warner -WORKS_FOR-> GitHub
- GitHub -PARTICIPATED_IN-> GitHub staff reduction announcement
- There’s something going on with AI startups in France -MENTIONS-> GitHub

Entities: GitHub, ChatGPT, TechCrunch, GPT-4, Early impressions of Google’s Gemini aren’t great, Filip Piekniewski, Mohsen Azimi, HTML, analog clock, Tic Tac
Toe game, Bard, Gemini Pro, Google, Gemini
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

Graph facts:
- One year later, ChatGPT is still alive and kicking -MENTIONS-> GPT-3.5
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> GPT-3.5
- How OpenAI's ChatGPT has changed the world in just a year -MENTIONS-> GPT-3.5
- GPT-4 gets a fine-tuning -MENTIONS-> GPT-3.5
- OpenAI -MADE-> GPT-3.5

Entities: GPT-3.5, TechCrunch, GPT-4, Early impressions of Google’s Gemini aren’t great, Bard, Gemini Pro, Robust Intelligence, AI security researchers, Google,
Gemini, Gemini Ultra
</item 4>

<item 5>
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

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, TechCrunch, GPT-4, RAG, 2023 Oscar winners, Early impressions of Google’s Gemini aren’t great, Bard, Gemini Pro, X, Google, OpenAI, Gemini,
generative AI
</item 5>

<item 6>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.

Graph facts:
- Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ -MENTIONS-> Google Search
- The people who ruined the internet -MENTIONS-> Google Search
- Is Google Search better than the rest? And is that fair? -MENTIONS-> Google Search
- The restaurant nearest Google -MENTIONS-> Google Search
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Google Search

Entities: Google Search, TechCrunch, Early impressions of Google’s Gemini aren’t great, Stross, Linux kernel, Benjamin Netter, Google News, Bard, Gemini Pro,
Google, Gemini, French
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
“When a user searches for information on a topic, Google displays a ‘Knowledge Panel’ to the right of the search results. This panel contains a summary of
content drawn from the Knowledge Graph database,” the complaint states. “Google compiled this massive database by extracting information from Publishers’
websites — what Google calls ‘materials shared across the web’ —and from ‘open source and licensed databases,'” it says.

By 2020, the Knowledge Graph had grown to 500 billion facts about 5 billion entities. But much of the “collective intelligence” that Google tapped into was
content “misappropriated from Publishers,” the complaint alleges.

Other Google technologies, like “Featured Snippets” where Google algorithmically extracts answers from webpages, were also cited as shifting traffic away from
publishers’ websites.

More importantly, perhaps, is the suit’s tackling of how AI will impact publishers’ businesses. The problem was recently detailed in a report on Thursday by The
Wall Street Journal, which led with a shocking statistic.

Graph facts:
- The inside story of Dave Clark's tumultuous last days at Flexport -MENTIONS-> The Wall Street Journal
- Europe’s artists and labels could lose $137m annually to US recorded music rightsholders if EU law isn’t changed, trade groups warn -MENTIONS-> Publishers
- A comprehensive list of 2023 tech layoffs -MENTIONS-> The Wall Street Journal
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> The Wall Street Journal
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet

Entities: Knowledge Graph, Publishers, Knowledge Panel, Featured Snippets, class action antitrust suit, misappropriation of Publishers’ content, The Wall Street
Journal, report on Thursday, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line, antitrust, TechCrunch, AI,
Google
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.

Graph facts:
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Helena World Chronicle
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> U.S. District Court in D.C.
- Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
-MENTIONS-> Washington Post
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Alphabet
- News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line -MENTIONS-> Search Generative Experience

Entities: SGE, Google Bard AI, GoogleBot, Google general search service, AdSense, News Media Alliance, Washington Post, Epic Games lawsuit against Google over
app store antitrust issues, app store antitrust issues, News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line,
class action antitrust suit against Google, Epic Games, Allen Institute for AI, TechCrunch, Google
</item 8>

<item 9>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -MENTIONS-> ChatGPT
- There’s something going on with AI startups in France -MENTIONS-> ChatGPT
- Founders, are events useful? -MENTIONS-> ChatGPT
- One year later, ChatGPT is still alive and kicking -MENTIONS-> ChatGPT
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> ChatGPT

Entities: ChatGPT, Stack Overflow, TechCrunch, Early impressions of Google’s Gemini aren’t great, war in Ukraine, Ukraine, Python, Bard, Gemini Pro, X, Google,
Gemini
</item 9>

<item 10>
Graph query row:
connected_entities: ['Google Nest', "200+ of the best deals from Amazon's Cyber Monday sale", 'Microsoft', None, 'antitrust lawsuit against Amazon']
relationships: ['MENTIONS', 'MENTIONS', 'FROM_CHUNK', 'FROM_CHUNK']
</item 10>
=== END ITEM id=a570a5579ef2ed5d ===

=== ITEM id=df3bc1654bfddb7c ===
Gold sentences and facts:
Sentence 0: Bankman-Fried was even compared to Warren Buffet and many called him the white horse of crypto (TechCrunch never did, for what it’s worth).
  Fact 0.0: Bankman-Fried was compared to Warren Buffet.
  Fact 0.1: Many people called Bankman-Fried the white horse of crypto.
  Fact 0.2: TechCrunch never compared Bankman-Fried to Warren Buffet or called him the white horse of crypto.
Sentence 1: The highly anticipated criminal trial for Sam Bankman-Fried, former CEO of bankrupt crypto exchange FTX, started Tuesday to determine whether he’s guilty of seven counts of fraud and conspiracy.
  Fact 1.0: The highly anticipated criminal trial of Sam Bankman-Fried started on a Tuesday.
  Fact 1.1: Sam Bankman-Fried is the former CEO of FTX.
  Fact 1.2: FTX is a crypto exchange.
  Fact 1.3: FTX is bankrupt.
  Fact 1.4: The trial determines whether Sam Bankman-Fried is guilty of seven counts of fraud and conspiracy.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution portrayed Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power, and influence.
  Fact 2.1: The defense argued that the FTX founder acted in good faith.
  Fact 2.2: The defense argued that the FTX founder never meant to commit fraud or steal.
  Fact 2.3: The defense argued that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
Sure, Sam Bankman-Fried’s trial has revealed that FTX was basically a seething mass of scams. But it has also revealed how Bankman-Fried blew up the lives of
the people who are closest to him.

Of course everyone who had money trapped on the FTX exchange when it collapsed — as the result of Bankman-Fried’s trading firm Alameda Research purloining its
customers’ funds — had their lives altered. Even if they are made whole by the bankruptcy proceedings, that’s still money that they should have had access to at
any time they wanted it. That may mean houses that weren’t purchased, medical procedures that were left undone, and life savings lost.

There is a simultaneous drama of ruined lives playing out in the courtroom

Bankman-Fried is charged with seven counts: two counts of wire fraud and five conspiracy charges. As we near the end of the prosecution’s case, I have been
struck by how bizarre it all is.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Alameda Research
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Alameda Research
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> Alameda Research
- The jury finally hears from Sam Bankman-Fried -MENTIONS-> Alameda Research
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Alameda Research

Entities: Alameda Research, bankruptcy proceedings, wire fraud, conspiracy charges, scams, In the end, the FTX trial was about the friends screwed along the
way, FTX trial, FTX, Sam Bankman-Fried, The Verge
</item 1>

<item 2>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Jacquelyn has been on the ground at the Southern District of New York courthouse, listening in to the trial in the same room as Bankman-Fried, so there was lots
to talk about.

TechCrunch+

TC+ subscribers get access to in-depth commentary, analysis and surveys — which you know if you’re already a subscriber. If you’re not, consider signing up.
Here are a few highlights from this week:

Inside the SBF trial: Rebecca and Jacquelyn report on the second day of the SBF and FTX trial. The prosecution painted Bankman-Fried as someone who knowingly
committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud
or steal and basically got in over his head.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> SBF Trial
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Google’s hardware event
- Founders, are events useful? -MENTIONS-> Pixel 8
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Pixel 8
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Pixel 8 Pro

Entities: teens’ startup, ad-free tier, Pixel 8, Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier, Jacquelyn,
Southern District of New York courthouse, Bankman-Fried, SBF and FTX trial, TechCrunch, TechCrunch+, the defense, FTX, Google, Rebecca, the prosecution
</item 2>

<item 3>
SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run (TechCrunch, 2023-10-02T13:11:07+00:00)
Come Tuesday, our eyes will be trained on the Manhattan Federal Court for the trial of Sam Bankman-Fried, the disgraced cryptocurrency entrepreneur accused of
orchestrating “one of the biggest financial frauds in American history”: the multibillion-dollar collapse of FTX, the crypto exchange he founded and led.

SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy. But, as we wait to hear both his version and the government’s version
of what happened, there will be another tale recounting how things played out: “Going Infinite,” a book by Michael Lewis chronicling the rise and fall of SBF,
is set to debut on the first day of the trial.

The episode has fired up a lot of people, indignant at what they feel was too-easy treatment for someone who many believe brought down the whole crypto house.

60 Minutes got rightly community noted here.

Graph facts:
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> Anna
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Going Infinite
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> SBF
- Tennis News: Kyrgios escalates beef with Becker, change ‘warrior’ Novak had to make, WTA criticised for charisma fail -MENTIONS-> 60 Minutes
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> 60 Minutes

Entities: Manhattan Federal Court, Going Infinite, 60 Minutes, SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run, Michael
Lewis, Sam Bankman-Fried trial, TechCrunch, conspiracy, FTX, Sam Bankman-Fried, Trump, fraud
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
“We had him going out directly to the internet,” LaVigne says. If he wrote things on Twitter or said things in interviews that weren’t true, that’s more fodder
for the government’s case. “They can point to that and say, ‘This is what he said, this is what actually happened.’”

Other evidence may include Signal messages and testimony from co-conspirators who plead guilty to their own charges. Alameda Research CEO Caroline Ellison, who
was also sometimes Bankman-Fried’s girlfriend, may play an important role — Bankman-Fried leaked her diaries to The New York Times and was consequently jailed
for witness tampering. The government has indicated FTX co-founder Gary Wang and engineering head Nishad Singh will also be among the witnesses called to give
testimony about Bankman-Fried.

Bankman-Fried’s defense can also introduce risks for people who dealt with him

There may be testimony from lenders, venture capitalists, and customers in order to establish the basis for some charges.

Graph facts:
- He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. -PUBLISHED_BY-> The New York Times
- How U.S. Women’s Soccer Team Lured Emma Hayes Away From Chelsea -PUBLISHED_BY-> The New York Times
- After a Stunning Snub, College Football Is Facing an Identity Crisis -PUBLISHED_BY-> The New York Times
- N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders -PUBLISHED_BY-> The New York Times
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -PUBLISHED_BY-> The New York Times

Entities: The New York Times, Gary Wang, Caroline Ellison, Alameda Research, defense, Nishad Singh, Signal, FTX trial, The FTX trial is bigger than Sam
Bankman-Fried, LaVigne, witness tampering, FTX, Sam Bankman-Fried, government, Twitter
</item 4>

<item 5>
Graph query row:
entity: Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand
relationship: PUBLISHED_BY
related: Fortune
</item 5>

<item 6>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.

Graph facts:
- Thane Rehn -PARTICIPATED_IN-> SBF Trial
- Is Sam Bankman-Fried’s defense even trying to win? -MENTIONS-> Thane Rehn
- Thane Rehn -SAID-> SBF
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> Thane Rehn
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> stealing customer funds

Entities: Thane Rehn, Larry David, stealing customer funds, Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first
witnesses take the stand, Alameda Research, Fortune, FTX, Sam Bankman-Fried, Tom Brady, government
</item 6>

<item 7>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
All the co-conspirators seem oddly younger than their years, as though they got stuck in the “math camp forever” environment Bankman-Fried created for them and
simply failed to grow up.

The prosecution has made a strong case that Bankman-Fried presided over the various forms of fraud that played out at Alameda Research and FTX, orchestrating
some of it and engaging in some of it directly. In the process of making this case, there is a simultaneous drama of ruined lives playing out in the courtroom.

Everyone who loved Bankman-Fried most in the world is now caught up in the case. We have:

Adam Yedidia, Bankman-Fried’s college roommate, who testified with immunity because he was concerned he may have unknowingly worked on code that contributed to
the fraud. “I love you Sam,” he texted at one point. “I’m not going anywhere.” That was before he realized FTX had stolen customer funds — at which point, he
resigned.

Graph facts:
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Bankman-Fried
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Bankman-Fried
- Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court -MENTIONS-> Bankman-Fried
- Today the FTX jury suffered through a code review -MENTIONS-> Bankman-Fried
- Bankman-Fried -SAID-> Alameda

Entities: Bankman-Fried, Adam Yedidia, Alameda Research, In the end, the FTX trial was about the friends screwed along the way, FTX trial, FTX, fraud, The Verge
</item 7>

<item 8>
SBF’s trial starts soon, but how did he — and FTX — get here? (TechCrunch, 2023-10-01T14:00:29+00:00)
U.S. Attorney Damian Williams called Bankman-Fried’s alleged crimes “one of the biggest financial frauds in American history,” in a press conference.

The aftermath

FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges
in relation to the FTX collapse. They’re also facing civil penalties from the U.S. Securities and Exchange Commission (SEC) and Commodity Futures Trading
Commission (CFTC) alongside the criminal charges. Wang and Ellison plan to cooperate with prosecutors and will be major witnesses in the trial, given their
close ties to Bankman-Fried, FTX and Alameda.

In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, conspiracy to commit money laundering and conspiracy to misuse customer
funds. He could face up to 115 years in jail if convicted on all charges.

Graph facts:
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> FTX collapse
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF’s trial
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> The New York Times
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> SBF
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> Series C round

Entities: FTX collapse, SBF’s trial starts soon, but how did he — and FTX — get here?, Damian Williams, Gary Wang, Caroline Ellison, Bankman-Fried’s trial, Wang
and Ellison pleaded guilty to federal criminal charges, Bankman-Fried pleaded not guilty, financial fraud, conspiracy to commit money laundering, conspiracy to
misuse customer funds, civil penalties, federal criminal charges, Alameda Research, TechCrunch
</item 8>

<item 9>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.

Graph facts:
- SBF’s trial starts soon, but how did he — and FTX — get here? -MENTIONS-> crypto
- SBF, riding high on FTX, reportedly offered $5B to Trump to cancel President re-run -MENTIONS-> crypto
- Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand -MENTIONS-> crypto
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> crypto
- SBF Trial: The latest updates from the FTX collapse’s courtroom drama -MENTIONS-> crypto

Entities: crypto, Alameda, FTX trial, second trial, misappropriated FTX customer deposits, political contributions, The FTX trial is bigger than Sam
Bankman-Fried, real estate, FTX, Sam Bankman-Fried, The Verge
</item 9>

<item 10>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
It is honestly kind of incredible to watch a man torpedo his own credibility on direct testimony. We’re not even at the cross yet, and the judge has already
instructed him to answer the question he’s being asked by his own lawyer.

The jury is watching all of this intently.

The main thing that’s been clear so far from Bankman-Fried’s testimony is that the man really loves the sound of his own voice. So far, the count of “Objection,
narrative!” to Bankman-Fried’s answers, followed by “Sustained” is at three.

Also, sometimes when Bankman-Fried says “we,” he only means himself.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, has been found guilty on seven counts including charges of wire fraud. FTX was a fraud
“from the start,” the Securities and Exchange Commission alleged — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer
funds.” Follow along for all the latest news and regular updates from the trial.

Graph facts:
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried was a terrible boyfriend -MENTIONS-> Securities and Exchange Commission
- How is it still getting worse for Sam Bankman-Fried? -MENTIONS-> Securities and Exchange Commission
- Coinbase rallies more than 60% in same month that FTX and Binance founders brace for prison -MENTIONS-> Securities and Exchange Commission
- Sam Bankman-Fried didn’t ask where the $8 billion went -MENTIONS-> misappropriation of customer funds

Entities: Securities and Exchange Commission, misappropriation of customer funds, The jury finally hears from Sam Bankman-Fried, wire fraud, FTX, Sam
Bankman-Fried, trial, The Verge
</item 10>
=== END ITEM id=df3bc1654bfddb7c ===
