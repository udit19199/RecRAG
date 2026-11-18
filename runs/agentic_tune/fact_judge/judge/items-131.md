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

=== ITEM id=1a42b103d092d549 ===
Gold sentences and facts:
Sentence 0: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 0.0: Google did not release the full model, Gemini Ultra.
  Fact 0.1: Google released only a 'lite' version called Gemini Pro.
Sentence 1: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 1.0: The Justice Department is focused on the deals Google makes.
  Fact 1.1: Google's deals include a deal with Apple.
  Fact 1.2: Google's deals include a deal with Samsung.
  Fact 1.3: Google's deals include a deal with Mozilla.
  Fact 1.4: Google's deals include deals with many others.
  Fact 1.5: The Justice Department is focused on the deals so that Google is the default search engine on practically every platform.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.
</item 2>

<item 3>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.
</item 3>

<item 4>
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
</item 4>

<item 5>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
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
</item 8>

<item 9>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 9>

<item 10>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 10>
=== END ITEM id=1a42b103d092d549 ===

=== ITEM id=b20a738ca89e257b ===
Gold sentences and facts:
Sentence 0: Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
  Fact 0.0: The speaker noted that time-restricted eating can also improve the quality of sleep.
  Fact 0.1: The speaker noted that better sleep can help the brain recover better.
Sentence 1: Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are foolproof.
  Fact 1.0: Today, the internet is littered with tips and tricks that promise a sound sleep.
  Fact 1.1: Not all of the sleep tips and tricks on the internet are effective.
  Fact 1.2: None of the sleep tips and tricks on the internet are foolproof.

Retrieved items:
<item 1>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
The symptoms of insomnia can last a long time; a longitudinal study conducted in the UK found that 69 per cent of respondents affected by the sleep disorder at
the start of the study still had insomnia 12 months later.

This may go to show how complex treating insomnia can be. Explaining the appeal of quick fixes, Dr Browning says, “In the short term you’re changing something,
so [they] can often [work] like a placebo. If hunger is keeping you awake, eating before bed will help. If your mattress is uncomfortable, sleeping on the floor
will help.” But the problem isn’t just that these methods don’t necessarily work – it’s that they could be used to exploit vulnerable insomniacs.

“I’m a sleep therapist, my doctorate is in insomnia and it’s very, very frustrating,” she says. “People who have insomnia are desperate.
</item 1>

<item 2>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The researchers hope that these findings will lead to human trials.

The goal is to obtain funding to start a pilot study of patients next year.

"Accessible interventions like this have great translational potential, as they are available and are usually very well-tolerated," Desplats said.

The goal is to obtain funding to start a pilot study of patients next year.

"This may have special importance for the elderly , as time-restricted eating does not require limitation in calories or change in the diet, but may offer
important benefits ranging from metabolic and sleep regulation to potential cognitive improvement."

TIME-RESTRICTED EATING NO MORE BENEFICIAL THAN CALORIC RESTRICTION IN OBESE PATIENTS, STUDY SAYS

Adopting a new dietary pattern should not be a replacement for medical care , however, as Desplats emphasized the importance of patients' discussions about
options with their clinicians.
</item 2>

<item 3>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
One major limitation of the study is that it uses animal models that only represent certain aspects of disease and "do not present comorbidities or other
clinical diseases associated with aging, which are very common in Alzheimer’s patients," noted Desplats.

"Furthermore, the study was done under strictly controlled conditions that did not resemble the variety of people’s habits and environments," she added.

THE 8 BIGGEST ALZHEIMER’S DISEASE MYTHS — AND THE TRUTHS BEHIND THEM

The researchers hope that the public recognizes the importance of the circadian clock in regulating many aspects of health .

"Sustaining healthy sleep and eating habits that are more synchronized with the natural environment are vital in a society that is constantly exposed to food,
light and shift work," said Desplats.

Previous studies have supported this association between fasting and reduced Alzheimer’s risk.
</item 3>

<item 4>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
The goal of the restricted eating is to restore the circadian rhythm, the researchers said, which could help counteract the numerous disruptions that
Alzheimer’s patients face when it comes to sleep and schedule-related confusion.

HIDDEN BELLY FAT COULD SIGNAL ALZHEIMER’S DISEASE RISK 15 YEARS BEFORE SYMPTOMS SHOW UP, STUDY FINDS

"Our study emphasizes the power of feeding timing in aligning the circadian clock and its impact on the brain," said senior study author Paula Desplats, PhD,
professor in the Department of Neurosciences at UC San Diego School of Medicine, in a statement to Fox News Digital.

"We were hopeful to see some improvements in pathology, but did not expect such profound effects in reducing plaques and inflammation and improving memory," she
went on.

"The effects of time-restricted feeding on lower amyloid pathology were also measurable in blood using markers found in the clinic, which was another important
finding."

The researchers hope that these findings will lead to human trials.
</item 4>

<item 5>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
Joseph Antoun, CEO and chairman of the nutritech company L-Nutra in California, was also involved in the USC study.

"In these studies, mice fed the fasting-mimicking diet showed improved cognitive performance, motor learning and both short- and long-term memory, and promoted
the growth of new neurons in the brain," he told Fox News Digital via email.

"Fasting-mimicking diets have the potential to drastically reduce many aging and disease risk factors by reducing neuroinflammation or superoxide production in
the brain."

Fasting can help the brain function in a myriad of ways, said Antoun.

"Fasting decreases pro-aging factors that are secreted after we eat, which can slow down the aging of the brain," he told Fox News Digital in an emailed
statement.

Time-restricted eating can also improve the quality of sleep, he noted, which can help the brain recover better.
</item 5>

<item 6>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Lavender oil. Eating kiwis. Masturbation. Everyone has their own “hacks” when it comes to falling asleep in 2023; the days of counting sheep are long gone.
Jennifer Aniston’s top tip for battling insomnia is to “sleep with my phone at least five feet away from me”, while Martha Stewart recommends focusing on your
breathing.
</item 6>

<item 7>
Fasting could reduce signs of Alzheimer’s disease, studies suggest: ‘Profound effects’ (FOX News - Health, 2023-12-06T11:25:40+00:00)
It’s been proven that what people eat can help prevent or slow Alzheimer’s disease — but what about when they eat?

Participating in intermittent (time-restricted) fasting could lead to a reduced risk of cognitive deterioration, a recent study published in the journal Cell
Metabolism suggests.

Researchers at University of California San Diego School of Medicine adjusted the feeding schedule of certain groups of mice so that they only ate within
six-hour windows each day.

ALZHEIMER’S BLOOD TEST COULD HIT THE MARKET IN EARLY 2024, RESEARCHERS SAY

(This would be equivalent to 14 hours of fasting for humans, the researchers noted.)

Compared to a control group of mice that ate on demand, the fasting mice showed improvements in memory, were less hyperactive in the evenings and had fewer
sleep disruptions .

They also were shown to have less buildup of amyloid proteins in the brain, which is a hallmark of Alzheimer’s disease.
</item 7>

<item 8>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
In her 1973 feminist novel Fear of Flying, author Erica Jong notes that, sometimes, the only way to overcome sleeplessness is by pretending you don’t care about
nodding off. “Then sleep became piqued, like a rejected lover, and crept up to try to seduce you.”

In a world where human sleep patterns have increasingly been plunged into chaos, people are having to turn to ever more creative – and often bizarre – means of
getting their 40 winks in. Today, the internet is littered with tips and tricks that promise a sound sleep, but not all of them are effective and none are
foolproof. (Kiwis, by the way, contain serotonin and antioxidants, which is why it’s thought they might have sleep-promoting benefits.) The result is a society
with a growing problem on its hands: persistent sleeplessness – otherwise known as insomnia.
</item 8>

<item 9>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“People who have insomnia are desperate. I treat patients who are suicidal, who have quit their jobs, whose lives are completely on hold because of their
sleeping issues. And they’re vulnerable. My clients have spent hundreds and hundreds of pounds on things that made no difference. If people are being mis-sold
expensive things and they’re being preyed upon, that’s a huge problem.”

Rosie Davies-Smith, a PR consultant, says she has spent more than £600 on a variety of treatments, including a combination of doctor-prescribed and
over-the-counter medications, teas, supplements, weighted blankets, and eye masks. “While supplements, exercise, and sleep hygiene provided some relief, the
true breakthrough came from addressing the root causes of my insomnia through therapy,” she tells me.
</item 9>

<item 10>
Sleepy tea, counting backwards and dense earplugs: Inside the lives of insomniacs (The Independent - Life and Style, 2023-10-01T08:55:56+00:00)
“At this point, you’re no longer worried about the traumatic incident; you’re now worried about not being able to sleep.” It’s a self-fulfilling prophecy.

Social media consultant and mother of two, Amy Fieldhouse-Downes, started suffering from insomnia after the birth of her second son in 2019. She believes the
affliction was rooted in anxiety. “If you Google how to go to sleep, I’ve probably tried everything on there,” she says. “Going to bed earlier, going to bed
later, meditations that would get cut off by an advert, lavender products, sleep cream, eye masks, Sominex sleep tablets. And I even used it as an excuse to
have a few glasses of wine.” It wasn’t until she tried sleeping with specialised earplugs that she was finally able to get some rest.
</item 10>
=== END ITEM id=b20a738ca89e257b ===

=== ITEM id=af0328fdced22c68 ===
Gold sentences and facts:
Sentence 0: But “all of that, all of it, was built on lies,” Rehn declared to the jury.
  Fact 0.0: Rehn declared to the jury that all of that was built on lies.
  Fact 0.1: Rehn declared to the jury that all of it was built on lies.
Sentence 1: FTX co-founder and former CTO Gary Wang, and Alameda Research’s former CEO, Caroline Ellison, both pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.0: Gary Wang is an FTX co-founder.
  Fact 1.1: Gary Wang is FTX's former CTO.
  Fact 1.2: Caroline Ellison is Alameda Research's former CEO.
  Fact 1.3: Gary Wang pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.
  Fact 1.4: Caroline Ellison pleaded guilty in December 2022 to federal criminal charges in relation to the FTX collapse.

Retrieved items:
<item 1>
Sex, Signal messages, and sabotaging FTX's code: SBF execs and Bahamas roommates tell all in court (Cnbc | World Business News Leader,
2023-10-06T20:17:00+00:00)
Ellison is expected to take the stand under a similar arrangement.

U.S. District Judge Lewis Kaplan presides as Gary Wang testifies during the fraud trial of Sam Bankman-Fried over the collapse of FTX, the bankrupt
cryptocurrency exchange, at Federal Court in New York City, U.S., October 6, 2023 in this courtroom sketch. Jane Rosenberg | Reuters

Born in China, Wang moved to the U.S. at age 7, and grew up in Minnesota before going to the Massachusetts Institute of Technology to study math and computer
science. He worked at Google after college. Wang, who first met Bankman-Fried during high school at a summer camp, owned 10% of Alameda, while his boss owned
the other 90%. Wang told the court about the advantages that Alameda received by having code baked into FTX's software that allowed special access to the crypto
exchange.
</item 1>

<item 2>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Alameda Research, a crypto hedge fund he also owned, argued Rehn. Using Caroline Ellison, his on-again off-again girlfriend and the CEO of Alameda, as a front,
Bankman-Fried had “secret access” to customer money—both cash and crypto—the government claimed.

Moreover, Bankman-Fried allegedly directed employees to conceal the flow of money into FTX’s coffers and forged financial documents distributed to lenders and
investors. “The defendant lied to the world,” Rehn alleged.

And who was this defendant? Not a crypto “boy” genius, as so many in the media (Fortune included) have written, but a “man” who “stole billions of dollars from
thousands of victims,” Rehn said. “You will see the full picture.”

…or a well-meaning founder?
</item 2>

<item 3>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Rather, he was willing to give up his personal wealth to make customers whole, Cohen argued.

“In the end, Sam started and built two billion-dollar businesses,” he concluded. “He didn’t steal any money.”

A Frenchman who lives in London testifies in New York

After lawyers from both sides depicted two very different Bankman-Frieds, the prosecution called its first two witnesses to the stand—and they weren’t
blockbuster names or former lieutenants-turned-government-cooperators, like Ellison.

The first was a victim: Marc-Antoine Julliard, a Paris-born cocoa trader who lives in London. In 2021, Julliard, who had coiffed hair and spoke with a strong
French accent, decided to invest in crypto and landed on FTX as his exchange of choice, where he traded cryptocurrencies like Bitcoin and Dogecoin.

On Nov. 8, in the crypto exchange’s final few days, he tried to pull out his cash and crypto.
</item 3>

<item 4>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
How much? Almost $100,000, he said. And was he ever able to? “Never,” he told prosecutors.

Shortly afterward, as the trial neared late afternoon, the government called Adam Yedidia to the stand. A fast-talking graduate of MIT, he and Bankman-Fried
were close friends in college, he said. And after Bankman-Fried left Jane Street, the high-frequency trading firm where the former billionaire got his start in
finance after MIT, he persuaded Yedidia to join him as a trader at Alameda and then as a developer at FTX.

When Yedidia first took the stand, Danielle Sassoon, one of the lead prosecutors, said that the college friend of Bankman-Fried had legal immunity during his
testimony. Why did he make such a deal with the government, she asked.

“I was concerned that I had unwittingly written code that contributed to a crime,” he said.
</item 4>

<item 5>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
FTX was no Ponzi scheme. It was a “very innovative, successful company.” And the business practices between the two were reasonable, he argued, claiming that
Alameda acted legally as an FTX customer, payment processor, and market maker, or financial entity that acts as a trading partner for customers looking to buy
and sell cryptocurrencies.

In an analogy he employed throughout his opening statement, he said that “working at a startup is like building a plane as you’re flying it” and that businesses
sometimes fail. In fact, he specifically pointed the finger at Ellison, the former CEO of Alameda, who, he said, did not adequately protect her hedge fund from
the inherent risk of the crypto markets.

When the walls came closing in and the aforementioned plane approached the “eye of the storm,” Bankman-Fried did not act like someone who was guilty. Rather, he
was willing to give up his personal wealth to make customers whole, Cohen argued.
</item 5>

<item 6>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“I was concerned that I had unwittingly written code that contributed to a crime,” he said.

Soon, however, the clock neared 4:30 p.m., and court adjourned for the day. Yedidia will continue his testimony on Thursday, followed by Matt Huang, a former
partner at the high-powered venture capital firm Sequoia Capital, and then Gary Wang, a key Bankman-Fried lieutenant and one of the government’s star witnesses.
</item 6>

<item 7>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“One year ago, it looked like the defendant was on the top of the world,” began Thane Rehn, a prosecutor for the government, in his opening statement. The
former CEO of FTX oversaw a supposedly thriving crypto exchange, jetted between international locales, and hobnobbed with celebrities like Tom Brady and Larry
David. He repeatedly emphasized to customers that their money was safe and secure.

But “all of that, all of it, was built on lies,” Rehn declared to the jury. “Behind the curtains, he was not what he appeared to be.” What followed was a
roughly 30-minute story that repeatedly emphasized how Bankman-Fried allegedly stole customer funds to facilitate his jet-setting lifestyle, donate millions to
political candidates, and finance risky bets.

The key to his alleged scheme? Alameda Research, a crypto hedge fund he also owned, argued Rehn.
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
</item 8>

<item 9>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
Who is Sam Bankman-Fried, the former CEO of crypto exchange FTX? Is he a liar and a fraud? Or did he act in good faith, and like many a startup founder, fly too
close to the sun?

That answer ultimately lies with the jury, which was selected on Wednesday morning before lawyers for the government and then Bankman-Fried swapped two very
different stories of the former crypto mogul’s sudden rise and almost instantaneous fall.

Here’s what happened on the second day of the trial, which featured pointed allegations, a friend from MIT, and an audience replete with big names, including
Bankman-Fried’s professorial parents and Damian Williams, U.S. attorney for the Southern District of New York.

A conscious criminal…

The prosecution’s account of the alleged crimes by Bankman-Fried, who spent most of his day in court staring into a laptop while seated between his two
attorneys, featured a study in contrasts.
</item 9>

<item 10>
Is Sam Bankman-Fried a bad ‘man’ or a good ‘boy’? Lawyers swap opening statements before first witnesses take the stand (Fortune, 2023-10-04T23:32:53+00:00)
“You will see the full picture.”

…or a well-meaning founder?

But Bankman-Fried, whose cheekbones were more prominent after spending about seven weeks in a Brooklyn prison, was no liar, according to Mark Cohen, one of his
lawyers. “Sam didn’t defraud anyone,” he said early on in his opening statement.

What the jury will see is a nerdy startup founder who acted in “good faith,” not the prosecution’s “cartoon of a villain.” (Cohen repeatedly harped on
Bankman-Fried’s allegedly good-faith actions throughout his address to the jury.)

Alameda was not subterranean or shady. It was a successful hedge fund, he said. FTX was no Ponzi scheme. It was a “very innovative, successful company.”
</item 10>
=== END ITEM id=af0328fdced22c68 ===

=== ITEM id=779d3986770f69fe ===
Gold sentences and facts:
Sentence 0: Monday’s trial hearing revealed plenty of juicy tidbits, including the $26.3 billion Google spent making itself the default search engine across platforms in 2021, how Google tried to take it further and have Chrome preinstalled on iPhones and more.
  Fact 0.0: Monday's trial hearing revealed plenty of juicy tidbits.
  Fact 0.1: Monday's trial hearing revealed that Google spent $26.3 billion making itself the default search engine across platforms in 2021.
  Fact 0.2: Monday's trial hearing revealed how Google tried to take it further.
  Fact 0.3: Monday's trial hearing revealed how Google tried to have Chrome preinstalled on iPhones.
  Fact 0.4: Monday's trial hearing revealed more tidbits besides these.
Sentence 1: In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones.
  Fact 1.0: In the last roundup, the author learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms.
  Fact 1.1: In the last roundup, the author learned how Google tried to have Chrome preinstalled on iPhones.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google siphons off news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google siphons off news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google siphons off news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 1>

<item 2>
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
</item 2>

<item 3>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 6>

<item 7>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 7>

<item 8>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 8>

<item 9>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 9>

<item 10>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Quick backstory: Yahoo inked a deal with Mozilla way back in 2014 to pay the browser maker $375 million annually to be the default search engine on Firefox. At
the time, Google was offering $276 million, said Baker. Reader, that was a bad deal.

“I felt strongly that Yahoo was not delivering the search experience we needed and had contracted for,” she said, according to Bloomberg. The executive said
Yahoo had promised to reduce the number of ads and offer less user tracking, but ended up gradually serving users more ads anyway.

“The number of users who stayed with Firefox declined noticeably during the years when Yahoo was the default,” said Baker.

Baker, who was providing a recorded deposition for Google’s defense, noted that Mozilla’s users apparently wanted and expected Google.

However, that is not the full story. Yahoo was already well behind Google in terms of search technology at that point.
</item 10>
=== END ITEM id=779d3986770f69fe ===

=== ITEM id=bece123351c69533 ===
Gold sentences and facts:
Sentence 0: The second week of the trial’s standout testimony came from Alameda Research’s former CEO Caroline Ellison, who claimed she took $14 billion from customers to repay debts to lenders, using them as a line of credit under the instruction of SBF.
  Fact 0.0: The second week of the trial's standout testimony came from Caroline Ellison.
  Fact 0.1: Caroline Ellison is Alameda Research's former CEO.
  Fact 0.2: Caroline Ellison claimed she took $14 billion from customers.
  Fact 0.3: Caroline Ellison claimed she took the $14 billion to repay debts to lenders.
  Fact 0.4: Caroline Ellison claimed she used customers as a line of credit under the instruction of SBF.
Sentence 1: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry.
  Fact 1.0: Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto.
  Fact 1.1: Bankman-Fried made himself out to be the trustworthy face of a sometimes-shady industry.
Sentence 2: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.
  Fact 2.0: The prosecution painted Bankman-Fried as someone who knowingly committed fraud.
  Fact 2.1: The prosecution said Bankman-Fried committed fraud to achieve great wealth, power and influence.
  Fact 2.2: The defense countered that the FTX founder acted in good faith.
  Fact 2.3: The defense countered that the FTX founder never meant to commit fraud or steal.
  Fact 2.4: The defense countered that the FTX founder basically got in over his head.

Retrieved items:
<item 1>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
I’ve got some shitty ex-boyfriends, but none of them made me the CEO of their sin-eater hedge fund while refusing to give me equity and bragging about how there
was a 5 percent chance they’d become the president of the United States, you know? Absolutely counting my blessings after Caroline Ellison’s first day on the
stand. I wonder how many of the nine women on the jury are doing the same.

Ellison was the head of Alameda Research, the aforementioned hedge fund, during the implosion of it and FTX. She’s already pleaded guilty to criminal charges
stemming from one of the worst romantic relationships I’ve ever heard of, and her testimony was widely anticipated before the trial. Today, that took the form
of discussing a damning spreadsheet — one she prepared for her ex and boss Sam Bankman-Fried, now the defendant in a criminal fraud trial.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy.
</item 1>

<item 2>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
Bankman-Fried is accused of using “billions of dollars in misappropriated FTX customer deposits” to help buy more than $200 million of real estate for himself,
make billions of dollars of investments for his own interest, and repay Alameda’s lenders, according to the indictment, Additionally, Bankman-Fried allegedly
used more than $100 million of customer funds to make political contributions; prosecutors can show evidence of those contributions in this trial, even though
they aren’t part of the charges brought. A second trial is scheduled for March 2024, with additional charges.

Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto — the trustworthy face of a sometimes-shady industry. He was also very interested
in publicity, sitting for many interviews both before and after the fall of FTX. The quick rise of FTX as an industry force was at least in part due to
Bankman-Fried’s appetite for attention.
</item 2>

<item 3>
The jury finally hears from Sam Bankman-Fried (The Verge, 2023-10-28T00:12:41+00:00)
Bankman-Fried said he worked 12 to 22 hours a day, and took one day off every couple of months. Because FTX had grown so much, he could no longer run both
companies, he said. Bankman-Fried handed the company off to Caroline Ellison and Sam Trabucco, who immediately after being named co-CEO promptly drifted away to
early retirement. (Quiet quitting king!) Bankman-Fried did remain involved in hedging and risk at Alameda, though.

About that $1 billion of revenue in 2021: Bankman-Fried definitely did not know that Singh, his employee, had backdated interest payments to get FTX “over the
line” to $1 billion. See, he’d just asked his employees to check and see if there was any source of funds that was missing to get to $1 billion. This testimony
was especially rambling.

Oh, also that MobileCoin loss? The one Wang said Alameda took to keep off FTX’s balance sheet?
</item 3>

<item 4>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
If it seems like his peer group supported him, or worse, fawned over him, that’s a real problem for the industry’s reputation.”

After FTX declared bankruptcy, Bankman-Fried went on an extensive media tour, doing interviews with The New York Times’ Andrew Ross Sorkin and Good Morning
America. Even under house arrest, he did interviews, including with The New Yorker. Bankman-Fried seemed to think that if he just explained things, everyone
would understand that what he did was a mistake, not a crime.
</item 4>

<item 5>
The FTX trial is bigger than Sam Bankman-Fried (The Verge, 2023-09-28T12:00:00+00:00)
According to the SEC, FTX was a fraud “from the start,” diverting customers’ funds to Alameda.

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.

“Really just old-fashioned embezzlement.”

What’s left of FTX is now being led by John J. Ray III — you may remember him as the guy who cleaned up Enron and who also said FTX is worse than Enron. Just
before the trial began, the FTX lawyers filed suit against Bankman-Fried’s parents, saying they should give back millions of dollars they got from their kid.
Ray has also referred to Bankman-Fried’s conduct as “really just old-fashioned embezzlement.”
</item 5>

<item 6>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Alameda’s credit line — which was taken in increments of $100,000 to $10 million at a time — was used for trading. Using the effectively unlimited funds
“allowed us to make profitable trades we couldn’t have made otherwise,” Ellison testified.

Customer funds were also used when Bankman-Fried bought back FTX shares from Binance, an early investor, in the summer of 2021. Bankman-Fried told Ellison it
was “really important,” otherwise “Binance would do things to mess with FTX.” Ellison says she told him Alameda didn’t have the money. So Bankman-Fried took $1
billion of FTX customer funds to buy out Binance, the first time Ellison recalled an amount that large. It was Bankman-Fried’s decision, she said, as he was the
CEO of FTX.

FTT was one of several “Sam coins”

There was also the FTT token, which was created by Bankman-Fried and Wang.
</item 6>

<item 7>
How is it still getting worse for Sam Bankman-Fried? (The Verge, 2023-10-12T23:08:14+00:00)
I have been waiting for a juicy cross-examination, as I live for chaos and drama. I am beginning to think I am not going to get one.

Ellison had given, in her direct testimony, fairly damning evidence tying FTX CEO Sam Bankman-Fried to the conspiracy to take FTX customer funds. There were
fake balance sheets, one of which was sent to crypto lender Genesis. After a Genesis representative received the balance sheet, he texted Ellison to tell her
he’d spoken to Bankman-Fried — strongly suggesting that Bankman-Fried was aware of the contents of the fake balance sheet. Not great!

Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for seven counts of wire fraud and conspiracy. FTX was a fraud “from the
start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of customer funds.” Follow
along for all the latest news and regular updates from the trial.
</item 7>

<item 8>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
FTX was a fraud “from the start,” the Securities and Exchange Commission alleges — with a “multi-billion-dollar deficiency caused by his own misappropriation of
customer funds.” Follow along for all the latest news and regular updates from the trial.

The day started off promisingly for the defense as it cross-examined Gary Wang, the chief technology officer of FTX and co-owner of both FTX and Alameda.
Christian Everdell, one of Bankman-Fried’s defense attorneys, couldn’t undo the damage of last week’s code review. But he managed to shake the rust off long
enough to make Wang sound less reliable, drowning the jury in confusing technicalities.

Last week, Wang testified that Alameda got access to a special credit line and an option to take its balance into the negative without triggering liquidation —
something he alleged other accounts at FTX didn’t get. Everdell tried to undermine this claim by talking about the spot margin program, which let users lend
each other assets for margin trading.
</item 8>

<item 9>
Sam Bankman-Fried was a terrible boyfriend (The Verge, 2023-10-10T23:50:21+00:00)
Using the effectively unlimited funds “allowed us to make profitable trades we couldn’t have made otherwise”

Bankman-Fried didn’t grant Ellison equity, even though she asked; he told her it would be too complicated. Instead, she got a $200,000 salary, even as CEO, and
bonuses twice a year, which ranged from $100,000 to $20 million.

Initially, Alameda and FTX were “very integrated,” Ellison said. They were run by the same team, from the same office. And when Alameda was scrounging for
funds, Bankman-Fried told Ellison that FTX would be a good source of capital. The $65 billion line of credit Alameda Research had meant that it did not have to
post collateral. There was no contract and no written terms, she testified. It also wasn’t visible to FTX’s auditors — she’d raised the question with
Bankman-Fried, and he told her not to worry about it.
</item 9>

<item 10>
In the end, the FTX trial was about the friends screwed along the way (The Verge, 2023-10-26T12:00:00+00:00)
That was before he realized FTX had stolen customer funds — at which point, he resigned.

Gary Wang, the co-founder of FTX and Alameda Research and another college roommate. He took the stand after Yedidia and sang like a canary about the secret
privileges he’d created for Alameda in the FTX code.

Nishad Singh, a longtime family friend, who copped to what sounded an awful lot like classic embezzlement. He said he’d backdated transactions on FTX to help
create doctored balance sheets to show to investors. He pleaded guilty to six charges, including wire fraud, money laundering, and conspiracy counts around his
political donations.

Caroline Ellison, Bankman-Fried’s ex-girlfriend. She told us about the seven phony balance sheets she’d created. She was clear: Alameda took customer funds;
Bankman-Fried knew about it and even invested them against her advice.

Joe Bankman, Bankman-Fried’s father.
</item 10>
=== END ITEM id=bece123351c69533 ===
