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

=== ITEM id=5e0ac915b0fcdcb4 ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The Justice Department focuses on these deals to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: In the last roundup, the writers learned that Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: In the last roundup, the writers learned that Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google CEO Sundar Pichai testified Monday in the U.S. government’s antitrust trial against the company. The executive defended Google’s business tactics,
including its deal with Apple and other partners to make Google the default search engine.

The lawsuit stems from a 2020 antitrust claim over Google’s dominance in the online search market. The claim is separate from one filed in January by the
Department of Justice, along with eight states including New York, California and Colorado, that aims to “halt Google’s anticompetitive scheme, unwind Google’s
monopolistic grip on the market and restore competition to digital advertising.”

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products.

Google argues that it’s just doing business. Everybody wants Google as the default engine because it’s the best. That doesn’t make its actions illegal, the
company says.
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 4>

<item 5>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 6>

<item 7>
The people who ruined the internet (The Verge, 2023-11-01T13:00:00+00:00)
Enter Google. The company revolutionized search by evaluating websites based on links from other websites, seeing each link as a vote of relevance and
trustworthiness. The founders pledged to be a neutral navigation system with no ads: just a clean white screen with a search box that would bring people off of
the Google landing page and out to a helpful website as seamlessly as possible. Users quickly decided this link-based sorting methodology was superior to the
existing search engines, and by the end of 1999, Google was handling the majority of online queries.

“I was basically just spamming Facebook with cars and articles about cars and sending traffic to banner ads, and that turned into $120,000 a month.”

Mapstead, like many of the early practitioners of SEO, figured out how to adapt. Almost as soon as Google took over, a secondary market emerged for links. For a
few hundred bucks, a firm in India or the Philippines could provide thousands of links from blog networks built entirely for that purpose.
</item 7>

<item 8>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 8>

<item 9>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 10>
=== END ITEM id=5e0ac915b0fcdcb4 ===

=== ITEM id=575fca3ee684597a ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: The deals Google makes include deals with Apple, Samsung, Mozilla, and many others.
  Fact 0.2: The Justice Department is focused on these deals to ensure Google is the default search engine on practically every platform.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: The last roundup reported how Google spent $26.3 billion in 2021 making Google the default search engine across platforms.
  Fact 1.1: The last roundup reported how Google tried to have Chrome preinstalled on iPhones.

Retrieved items:
<item 1>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
This is where Bellshaw began to press on how exactly Apple decided Google had the best product. She asked Cue if Apple believes user privacy is important, to
which he said, “Absolutely.” Then, she showed a series of emails and slides in which Cue and Apple railed against Google’s privacy policies. Cue readily agreed.
“We’ve always thought we had better privacy than Google,” he told Bellshaw. He said that one provision of the ISA with Google was that Google had to allow
people to search without logging in and that Apple has done things in Safari and around its platforms to make it harder for Google or anyone else to track
users.

Bellshaw never quite said it, but the DOJ’s implication seemed to be that, essentially, Google is a privacy menace anathema to everything Apple believes is
important to its users, but Apple gives it a central place in its platform because Google pays it so handsomely. Bellshaw asked Cue to review some of Apple’s
financial filings.
</item 1>

<item 2>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
He said there still isn’t one.

That question — whether Apple picked Google because it’s the most lucrative choice or the best product — was a key part of Cue’s testimony and, in fact, a key
part of the DOJ’s entire case against Google. The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and
many others — to ensure it is the default search engine on practically every platform.

Bellshaw asked Cue a number of questions about the iPhone setup process. Those three screenshots showed the Appearance screen that shows up when you first boot
up your iPhone so you can pick font sizes; the location-tracking prompt that appears when you open Maps; and the App Tracking Transparency pop-up that tells you
when an app wants to collect your data. Cue objected to all these things being considered part of setup, but Bellshaw’s point was that Apple offers its users a
choice about lots of things, big and small, and that search could be one of them.
</item 2>

<item 3>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Bellshaw asked Cue to review some of Apple’s financial filings. Isn’t it true that the ISA represents a significant portion of Apple’s profits, she asked? Cue
said that’s not how Apple looks at it because it doesn’t account for all the work Apple did to make its platform so appealing that an agreement like this could
work as well as it does.

Later, after a closed session in the courtroom and a break for lunch, Google lawyer John Schmidtlein led Cue through a history of the Google / Apple
partnership, and a history of the Safari browser. Cue noted that Safari’s combination of URL and search bar was a user interface innovation, and the seamless
Google integration was part of what made it work. In early promotional materials for Safari, Schmidtlein pointed out, the Google integration was nearly always
mentioned.
</item 3>

<item 4>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 4>

<item 5>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
That doesn’t make its actions illegal, the company says.

Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in
2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.

Google paid $26 billion in 2021 to be everyone’s default search engine

When Google’s search head Prabhakar Raghavan testified in court on October 28, he revealed that the tech giant had paid $26.3 billion in 2021 to multiple
browsers, phones and platforms, from companies including Apple, Samsung and Mozilla, The Verge reports.

About $18 billion of that total amount went directly to Apple, according to a New York Times report published earlier this month. Google has had its deal with
Apple in place since 2003.

When questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified
that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO
Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.

Google agreed not to promote Chrome to Safari users

As part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with
banners, pop-ups and other annoying means in other Google apps.

The agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.

Google tried to be preinstalled on iPhones

Pichai admitted to attempting to get Tim Cook to preinstall Google on every iOS device back in 2018, according to The Verge.
</item 6>

<item 7>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 7>

<item 8>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Cue is one of the highest-profile witnesses in the case so far, in part because the deal between Google and Apple — which makes Google the default search engine
on all Apple devices and pays Apple billions of dollars a year — is central to the US Department of Justice’s case against Google.

Cue had two messages: Apple believes in protecting its users’ privacy, and it also believes in Google. Whether those two statements can be simultaneously true
became the question of the day.

Apple is in court because of something called the Information Services Agreement, or ISA: a deal that makes Google’s search engine the default on Apple’s
products. The ISA has been in place since 2002, but Cue was responsible for negotiating its current iteration with Google CEO Sundar Pichai in 2016. In
testimony today, the Justice Department grilled Cue about the specifics of the deal.

When the two sides renegotiated, Cue said on the stand, Apple wanted a higher percentage of the revenue Google made from Apple users it directed toward the
search engine.
</item 8>

<item 9>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
Discussion of specific numbers was reserved for closed court sessions, but Cue wanted Apple to get a higher percentage, while Pichai wanted to keep the deal as
it was. They eventually compromised on some other number we weren’t told in court, and Google has been paying Apple that amount since.

“I always felt like it was in Google’s best interest, and our best interest, to get a deal done.”

Meagan Bellshaw, a Justice Department lawyer, asked Cue if he would have walked away from the deal if the two sides couldn’t agree on a revenue-share figure.
Cue said he’d never really considered that an option: “I always felt like it was in Google’s best interest, and our best interest, to get a deal done.” Cue also
argued that the deal was about more than economics and that Apple never seriously considered switching to another provider or building its own search product.
“Certainly there wasn’t a valid alternative to Google at the time,” Cue said. He said there still isn’t one.
</item 9>

<item 10>
Apple defends Google Search deal in court: ‘There wasn’t a valid alternative’ (The Verge, 2023-09-26T17:01:22+00:00)
But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the
biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple
would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.
</item 10>
=== END ITEM id=575fca3ee684597a ===

=== ITEM id=1172a167eedc1a18 ===
Gold sentences and facts:
Sentence 0: This isn’t the first time that Jada has opened up about her long-term commitment to her husband.
  Fact 0.0: This is not the first time Jada has opened up about her long-term commitment to her husband.
Sentence 1: Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus.
  Fact 1.0: Jada praised her children for how they helped her grow.
  Fact 1.1: Jada said: 'My children, they're little gurus.'
Sentence 2: She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep sense of self-acceptance”.
  Fact 2.0: Jada and Will have two children.
  Fact 2.1: Jada and Will's children are Jaden, who is 25, and Willow, who is 22.
  Fact 2.2: Jada praised her children for being her 'little gurus' throughout this time.
  Fact 2.3: Jada's children taught her 'a deep sense of self-acceptance'.

Retrieved items:
<item 1>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
During an interview with Hoda Kotb for Today, which aired on 13 October, the actor confirmed that she and her partner have been living “completely separate
lives” for the past seven years. However, according to Jada, their choice to live in separate homes doesn’t mean that they’ll be getting divorced anytime soon.

“I made a promise that there will never be a reason for us to get a divorce,” she said. “We will work through whatever. And I just haven’t been able to break
that promise.”

When asked why she and Will didn’t “just amicably divorce,” Jada simply replied: “We don’t want to.”

She then explained how her family shaped that decision, as she and the King Richard star share two children - Jaden, 25, and Willow, 22. Will also has a
30-year-old son, Trey, who he shares with ex-wife Sheree Zampino. “We love our family and we love each other.
</item 1>

<item 2>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Will Smith has spoken out about his “tumultuous” relationship with his wife, Jada Pinkett Smith, while speaking to a crowd of fans in Baltimore, Maryland.

The 55-year-old actor took the stage at the Enoch Pratt Free Library on 18 October, as he made an appearance at a press event for his wife’s book, Worthy,
alongside their two children – Jaden, 25, and Willow, 22.
</item 2>

<item 3>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Less than two weeks later, Jada and Will appeared on an episode of Jada’s Red Table Talk where Jada confirmed that she did have a relationship with Alsina,
saying that she and Will were separated at the time.

“We decided that we were going to separate for a period of time and you go figure out how to make yourself happy and I’ll figure out how to make myself happy,”
Jada said at the time.

“From there, as time went on, I got into a different kind of entanglement with August... Yes, it was a relationship absolutely.”

Jada also said that Will didn’t give permission, as the only person who could give her permission was herself.

2021: Jada discusses her and Will’s sex life, Will’s memoir is released

In another episode of Red Table Talk in October 2021, Jada said it “takes work” to maintain a spark in the bedroom.

“It’s hard... the thing Will and I talk about a lot is the journey.
</item 3>

<item 4>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed why she and her husband Will Smith never signed a prenuptial agreement before getting married.

The former talk show host, 52, spoke candidly about her husband during a recent interview with Parade. While discussing her new memoir, Worthy, Jada revealed
that she and Will didn’t sign a prenup when they were married in 1997 because they knew divorce wasn’t in the cards for them.
</item 4>

<item 5>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
Prior to sharing the video, Smith showed his support for his partner and her success in a statement toThe New York Times. During the interview, the outlet noted
that Smith said his wife’s memoir “kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” he added.

In an interview with People published on 10 October, Jada first revealed that she and her husband were living separately. “We’ve been doing some really
heavy-duty work together,” she said. “We just got deep love for each other and we are going to figure out what that looks like for us.”
</item 5>

<item 6>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.
</item 6>

<item 7>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Jada Pinkett Smith has revealed that she and husband Will Smith have been separated for the last seven years.

The Girls Trip star, 52, shared in a new interview with People published on 11 October that she and the King Richard actor, 55, are “still figuring it out” when
it comes to their marriage of 26 years. She explained at the time of the 2022 Academy Awards, when Will slapped Chris Rock, they had been separated for six
years.
</item 7>

<item 8>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
“We’ve been doing some really heavy-duty work together,” Jada said. “We just got deep love for each other and we are going to figure out what that looks like
for us.”

The pair have been married since 1997 and share son Jaden, 25, and daughter Willow, 22. The Fresh Prince of Bel-Air star also has a son Trey, 30, whom he shares
with his first wife, Sheree Zampino.

As for her children, Jada praised them for how they’ve helped her grow. “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance,”
she told the outlet.

Jada and Will have often opened up about their relationship over the years, and have had very public ups and downs throughout their marriage. Here’s a timeline
of their relationship.
</item 8>

<item 9>
Jada Pinkett Smith and Will Smith have been separated since 2016 - when should couples just get divorced? (The Independent - Life and Style,
2023-10-16T13:41:30+00:00)
In an interview with People on 10 October, Jada shared that during the 2022 Academy Awards - when Will slapped Chris Rock after he cracked a joke about Jada’s
appearance - the pair had actually been separated for six years. “We’ve been doing some really heavy-duty work together,” she explained, while promoting her new
memoir, Worthy. “We just got deep love for each other and we are going to figure out what that looks like for us.”

She praised her and Will’s two children - Jaden, 25, and Willow, 22 - for being her “little gurus” throughout this time, noting that they’ve taught her “a deep
sense of self-acceptance”. Although the married couple are not living together, she insisted that their marriage has never been an open one, as she and Will
have “eliminated the chance of betrayal” while going on to live their “separate lives”.
</item 9>

<item 10>
Jada Pinkett Smith reveals why she and Will Smith never signed a prenup (The Independent - Life and Style, 2023-10-16T15:48:10+00:00)
Jada then praised her children for how they’ve helped her grow, adding: “My children, they’re little gurus. They’ve taught me a deep sense of self-acceptance.”

On 16 October, Will broke his silence on his wife’s comments during an interview with the New York Times. The outlet noted that he said his partner’s memoir
“kind of woke him up” and that he has now realised she is more “resilient, clever and compassionate than he’d understood”.

“When you’ve been with someone for more than half of your life, a sort of emotional blindness sets in, and you can all too easily lose your sensitivity to their
hidden nuances and subtle beauties,” Will added.
</item 10>
=== END ITEM id=1172a167eedc1a18 ===

=== ITEM id=705d3f5af05aac1f ===
Gold sentences and facts:
Sentence 0: In blog posts and press materials, Google touted Gemini’s superior architecture and capabilities, claiming that the model meets or exceeds the performance of other leading gen AI models like OpenAI’s GPT-4.
  Fact 0.0: Google touted Gemini's superior architecture and capabilities in blog posts and press materials.
  Fact 0.1: Google claimed that Gemini meets or exceeds the performance of other leading gen AI models.
  Fact 0.2: OpenAI's GPT-4 is one of the other leading gen AI models Google named.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
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
</item 1>

<item 2>
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
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
But Google denies pushing users towards the most monetisable subjects.

Similarly, it’s also pretty common to see people complaining on the internet with some variation of “Google Search was a lot better 10 years ago”, which could
absolutely be true. But it would be a stretch to assume that’s because of the company’s monetisation practices, when there’s a much simpler explanation. In
2013, people were creating around 9 ZB of data every year (that’s nine zettabytes, or nine trillion gigabytes), which Google had to index constantly, whereas in
2023 it’s more like 120 ZB. And not a whole lot of that is static websites like it was a decade ago, it’s live information that may be nested within various
services and apps. What really happens when we search? Google has published extensive information about its Search procedures, so it’s not difficult to get its
own take on what happens in the background when you send off a query.
</item 6>

<item 7>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 8>

<item 9>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 10>
=== END ITEM id=705d3f5af05aac1f ===

=== ITEM id=c33d825cd9cedc2f ===
Gold sentences and facts:
Sentence 0: This whole line of argument is called “market definition.” If the court decides the relevant market is phones and app stores, not specifically Android ones, then Google’s on the road to victory.
  Fact 0.0: The whole line of argument is called 'market definition.'
  Fact 0.1: If the court decides the relevant market is phones and app stores, rather than specifically Android ones, Google is on the road to victory.
Sentence 1: She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue.
  Fact 1.0: The woman admits that she may have misinterpreted the evidence.
  Fact 1.1: The woman maintains that Google manipulates Search to maximise ad revenue.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content.
  Fact 2.2: The case argues that Google siphons off news publishers' readers.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue.
  Fact 2.4: The case argues that Google does this through anticompetitive means.

Retrieved items:
<item 1>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 1>

<item 2>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 2>

<item 3>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 3>

<item 4>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
The case also arrives alongside the filing of the U.S. Justice Department’s lawsuit against Google for monopolizing digital ad technologies, and references the
2020 Justice Department’s civil antitrust suit over search and search advertising (which are different markets from digital ad technologies in the more recent
suit).

“The anticompetitive effects of Google’s scheme cause profound harm to competition, to consumers, to labor, and to a democratic free press,” reads an
announcement posted to the website of the law firm handling the case, Hausfeld.

“Plaintiff Helena World Chronicle, LLC invokes the Sherman Act and Clayton Act to seek class-wide monetary and injunctive relief to restore and ensure
competition for digital news and reference publishing and set up guardrails to preserve a free marketplace of ideas in the new era of artificial intelligence,”
it states.

A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless.
</item 5>

<item 6>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
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
</item 7>

<item 8>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 8>

<item 9>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
SGE, it argues, offers web searchers a way to seek information in a conversational mode, but ultimately keeps users in Google’s “walled garden” as it
“plagiarizes” their content. Publishers also can’t block SGE because it uses the same web crawler as Google’s general search service, GoogleBot.

Plus, it says Google’s Bard AI was trained on a dataset that included “news, magazine and digital publications,” citing both a 2023 report from the News Media
Alliance and a Washington Post article about AI training data for reference. (The Post, which worked with researchers at the Allen Institute for AI, had found
that News and Media sites were the third largest category of AI training data.)

The case points to other concerns, too, like changing AdSense rates and evidence of improper spoliation of evidence on Google’s part, by its destruction of chat
messages — an issue raised in the recent Epic Games lawsuit against Google over app store antitrust issues, which Epic won.
</item 9>

<item 10>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
These are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being
anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.

What does it all mean?

The case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the
1990s.

If the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and
share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.

Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more
users. The DOJ has said that Google gets 16 times more data than Bing does everyday.
</item 10>
=== END ITEM id=c33d825cd9cedc2f ===
