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

=== ITEM id=9a4fbb4b5b2bbbd8 ===
Gold sentences and facts:
Sentence 0: Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.
  Fact 0.0: The Fortnite maker announced on September 28 that it is laying off 16% of its workforce.
  Fact 0.1: The layoff amounts to 870 people.
Sentence 1: Because while Google spent most of its first day attempting to explain complicated ins and outs of business, Epic was able to paint a black-and-white picture of good and evil with itself as the clear underdog.
  Fact 1.0: Google spent most of its first day attempting to explain complicated ins and outs of business.
  Fact 1.1: Epic was able to paint a black-and-white picture of good and evil.
  Fact 1.2: In that picture Epic itself was the clear underdog.
  Fact 1.3: Epic was able to do this because Google spent most of its first day on those explanations.

Retrieved items:
<item 1>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
In October 2019, Epic internally said it might pursue an aggressive pursuit model with the Epic Games Store, where it would aggressively pay more for exclusive
games and really pull in more and more users. Are you in the aggressive pursuit model?

No, we’re in a different model, which is called the “Ridiculously aggressive pursuit model.” If it’s a 10-slide deck, that’s on slide 11.

Did you get a bingo on your Verge bingo card?

Yeah, I think it was almost a complete shutout. I think there was only one cell in the entire board that was missing. I was really impressed — the thing for
that bingo card that really impressed me was “lick the cookie.” Do you remember? You may not even be old enough, but “lick the cookie” was in the 1999 US v.
Microsoft antitrust trial.
</item 1>

<item 2>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
You know, we expanded the company to nine times the size we were before Fortnite took off. And so we’re just trying to forge the strategy where we’re living
within our means and doing everything we absolutely can.

But we’ve gotten so much traction with Fortnite in so many different ways, and with the Epic Games Store, right, we’re going to continue investing heavily here.
The Epic Games Store is the untold success story in the background. Since it launched in 2018, we now have 80 million monthly active users. Steam has 120
million, so we’re catching them fast! Fortnite has more concurrent users right now than all Steam games combined. You should really expect dramatic changes that
benefit all developers over the next few years.

[Editor’s note: Sweeney isn’t right about Fortnite having more concurrent users than Steam. Epic spokesperson Natalie Muñoz confirmed Sweeney misinterpreted a
Polygon story about how Fortnite hit 7.6 million players at a time when the entirety of Steam had 10.1 million.
</item 2>

<item 3>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
I don’t know if that was on behalf of you and fellow developers. Can you tell us what that email was actually about?

Google didn’t leak the email; they leaked their ridiculously biased summary of it to Abner Li at 9to5Google. Don Harrison testified at trial that he didn’t
think Google leaked to the press. And then he was presented with a document including Google’s press team and Sameer Samat, head of all Android, basically
summarizing the articles that appeared as a result of Google’s leaking of our plans to the press.

So I was wondering if you could tell me what the email actually said.

Oh, it’s in evidence! This was the one I mentioned! Right before Christmas-ish 2019, we were planning to launch the new Marvel season with a bunch of Star Wars
content and its awesome lineup. I asked Google’s executives to let Fortnite come to the Google Play Store using our own payment method and to open up, you know,
using their own payment methods to all developers, not just Epic.
</item 3>

<item 4>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 4>

<item 5>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Well… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company
didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than,
say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.

But she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their
own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.

How can Epic possibly have a case against Google when Apple already won?

Epic declined to answer this on the record, among other questions we asked... but three things to consider:

1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.
</item 5>

<item 6>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A comprehensive list of 2023 tech layoffs From major layoffs at Google, Amazon and Microsoft to small fintech startups and apps

Last year’s techwide reckoning continues. The tech industry has seen more than 240,000 jobs lost in 2023, a total that’s already 50% higher than last year and
growing. Earlier this year, mass workforce reductions were driven by the biggest names in tech like Google, Amazon, Microsoft, Yahoo, Meta and Zoom. Startups
across many sectors also announced cutbacks through the first half of the year. And while tech layoffs slowed down in the summer and fall, it appears that cuts
are ramping up yet again.

Many economists have cautioned against fears of a recession, which would seem like a reason for optimism. But the momentum for a tech sector rebound has been
slow to build, resulting in tech companies continuing to cut back on their workforces and pivot from a growth mindset to one based on efficiency in the face of
stubborn market conditions.
</item 6>

<item 7>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Also, that one’s not over till the Supreme Court weighs in or declines to take a look.

2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!

3) Oh, and by the way, this is a jury trial.

Wait, why does that matter?

Epic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)

Maybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping
security holes will sway a jury against Epic instead? Who knows!

(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)

Didn’t other parties sue Google too?

They did! And then, they all settled.
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
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Hippo

Plans to eliminate 120 roles, the company said in an SEC filing October 30.

Liberty Mutual

Plans to cut 850 members of its staff, the insurance company confirmed October 27.

Salsify

Cut 110 jobs and are moving some roles abroad, CEO and co-founder Jason Purcell announced October 25.

Slync

Is shutting down, months after its former CEO Chris Kirchner was arrested on fraud charges and sued the company to pay for his legal bills.

Shipt

Announced October 24 in a company statement that it laid off 3.5% of their workforce and will close many open roles.

Pebble

The X alternative previously known as T2 announced October 24 that it is shutting down its operations.

Nokia

Announced October 19 that it will cut up to 14,000 jobs, or 16% of its workforce, following a sizable dip in Q3.

Roblox China

Cut 15 roles on October 24, two years after the joint effort with Tencent abruptly paused its service.
</item 9>

<item 10>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Enough that the judge has decided to explicitly tell the jury we’re not seeing all the evidence, because Google employees all the way up to CEO Sundar Pichai
were caught setting those conversations to auto-delete. This issue has come up in the US v. Google antitrust trial in Washington, DC — still ongoing at the
start of Epic’s trial — as well.

And the dirt isn’t all on Google’s side. Epic was caught straight-up admitting it premeditated the whole plan to bypass Google’s payment systems. Since Google
is countersuing over that, it could be a big deal.

When you say “straight-up admitting...”

Here is an email from Epic CEO Tim Sweeney that, I think, speaks for itself:

“Here is our official plan for communicating with Google about bypassing the Google Play Store: SAY NOTHING TILL IT SHIPS” Image via Epic v. Apple

We’ll have a whole cast of intriguing people called as witnesses during the trial who might dish even more.

Who will Epic and Google call as witnesses?
</item 10>

<item 11>
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
</item 11>

<item 12>
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
</item 12>

<item 13>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This wasn’t the only layoff round of the year. Unity lowered its headcount three times in 2023, affecting about 900 jobs. In its quarterly financial results in
November, Unity reported a yearly revenue increase of 69 percent and it told investors, “We continued to manage costs well.”

Sony cut 100 jobs at Bungie, a company it bought for $3.6 billion in 2022. According to developers that are still there, Sony executives are attempting to use
this upheaval to wrest more control of the studio from Bungie founders and leaders.

Epic Games fired roughly 830 people this year, or 16 percent of its staff. This included significant job cuts at Mediatonic, the studio behind Fall Guys that
Epic purchased in 2021.

Mediatonic

“For a while now, we've been spending way more money than we earn,” CEO Tim Sweeney wrote about the layoffs.
</item 13>

<item 14>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
Thank you for being here with us. It’s been a very engaging trial to watch. I’ve been there every day of the trial, and you’ve been there every day save one. So
my first question is why did you personally attend this trial every day save one — and what the heck happened on that one day?

Yeah, Epic is asking a lot of the court system and the jury here, spending four weeks on a major antitrust trial full of complicated facts and evidence. It
wouldn’t be right to start something like this and not show up. So I had to do that. And, you know, Phil Schiller sat throughout the entire Epic v. Apple trial,
as did I, so I think it’s just necessary to show respect for the legal process.

What happened on that one day you weren’t here?

Oh — sorry I can’t share, but there were some... non-Epic issues I had to deal with.

So this trial has been four years in the making.
</item 14>

<item 15>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
On Monday, Fortnite publisher Epic Games will drag a tech giant into court, alleging that its mobile app store is an illegal monopoly. You may be feeling some
déjà vu!

After all, didn’t Epic already go to trial with Apple, resulting in a ruling where Apple (mostly) won? Didn’t Epic fail to #FreeFortnite with its Llamacorn
legal gambit? Didn’t that all happen years ago? What are we doing here again?

The short answer: while Epic’s antitrust claims against Apple got their day in court, a similar lawsuit against Google never did. On November 6th, Epic v.
Google will finally go to trial... a mere 1,180 days after Epic originally sued.

Hi, I’m Sean, and I’ll be your guide to this whole delightful mess.

Update, December 15th, 2023: It’s basically over now. What happened? Read this, this, and this to get the whole picture.

Original story continues:

Llamacorn what now? What is happening?
</item 15>

<item 16>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
[Editor’s note: Every journalist in the courtroom agreed the jury was attentive; most were also surprised by the rapid verdict.]

They got it, they got it quickly, and they were able to pull apart what was going on and contrast it with the fiction of the story that Google was trying to
tell.

What was something that surprised you in the courtroom, something you heard come to light that you hadn’t heard before?

I’d suspected a lot of the practices that Google had, you know, since 2018 or so when we first started this, to such an extent that some folks would
occasionally call me a conspiracy theorist. It was really, really interesting to see that my understandings of what Google was doing behind the scenes were
actually true — you’re leaking our conversations to reporters to get negative stories written about us; you’re paying other developers off to convince them not
to launch their own stores; they were going around and paying carriers and OEMs secretly not to carry competing stores.
</item 16>

<item 17>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
It will note this process is streamlined and helps keep users safe. But Epic will claim that this process is made to be overly difficult for end users, with
multiple steps involved and “scare screens” that warn users of the security risks involved with sideloading.

Epic will also suggest that Google hid items from discovery by allowing execs and employees to have “off the record” chats, but Google will counter this as
well, saying it has provided the court with thousands of chat logs and millions of documents.

The tech giant will ultimately push the notion that this case isn’t about competition, but about money — that is, Epic Games wants to reach the Play Store’s 2.5
billion users without having to pay.

“Android’s choice and flexibility work well for consumers and developers of all sizes.
</item 17>

<item 18>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
September 2023

IronNet

In a regulatory filing published September 29, the cybersecurity startup has ceased all business activities as it prepares for Chapter 7 bankruptcy and has laid
off its remaining staff.

Naughty Dog

Is reportedly cutting at least 25 contract developers, two sources familiar with the situation told Kotaku.

Epic Games

Announced on September 28 that the Fortnite maker is laying off 16% of its workforce, amounting to 870 people.

Talkdesk

An impacted employee confirmed on September 27 that at least 140 people, if not more, were affected by Talkdesk’s third round of layoffs in less than 14 months.

Byju’s

Announced on September 26, the company plans to cut 5,000 jobs due to restructuring efforts. The startup has eliminated more than 10,000 positions in the past
two years.

Roblox

Confirmed to TechCrunch on September 21, Roblox let go of 30 employees in its talent acquisition organization. No other teams are impacted.
</item 18>

<item 19>
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
</item 19>

<item 20>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What were your settlement talks with Google CEO Sundar Pichai like?

We can’t talk about content, but we met for an hour and had a professional discussion... in which we didn’t reach a settlement. We were rather far apart, let’s
say, because what Epic wants ultimately is free competition and fair competition for everybody, and the removal of the payments tie and removal of the
anticompetitive measures, which obviously leads to far better deals for consumers and developers.

What we don’t want is a special deal just for ourselves, and Google’s strategy so far as you’ve seen from all these deals has been to basically take out all
their potential competitors one at a time while they’re still weak and small, before they build up or unite into a formidable force. Epic’s not going to go
along with something like that and do a special deal just for ourselves.

You’ve said that a couple times, but the email that you accused Google of leaking suggested that Epic was interested in a special deal of some sort at that
time.
</item 20>

<item 21>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
He continued, “I had long been optimistic that we could power through this transition without layoffs, but in retrospect I see that this was unrealistic.”

Electronic Arts was one of the first video game companies to institute significant layoffs this year, with a reduction of 6 percent of its workforce, or about
800 employees, in March. EA later cut jobs at Dirt and F1 studio Codemasters, which it purchased in 2021 for $1.2 billion. EA culled an estimated 1,130 jobs in
2023.

CD Projekt RED and Sega each laid off about 100 people in the past 12 months, while Ubisoft fired an estimated 255 employees. Microsoft cut 10,000 jobs across
its businesses early in the year, and that included about 100 people at Halo studio 343 industries.

343 industries

These are just some of the biggest names in layoffs in 2023.
</item 21>

<item 22>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We know that because they came to us and told us they were doing that. And we know it because after all that went down, I talked to an Activision employee who
was close to that effort, working with the team building the store, and he reported that the entire store team Activision had employed to build their competing
store had been disbanded as soon as Activision signed their deal with Google.

[Editor’s note: We did not hear from such a person at trial; I’d be eager to speak to them now!]

We know Activision was telling Google they were building their computing store; we know in Google internal discussions they said they didn’t want Activision
building a competing store. They agreed to sign this deal, and they were gleeful about the fact they dissuaded perhaps their number one competitor at the time
from launching their own store.

Four years later, Microsoft acquired Activision Blizzard — and one of the big talking points from Microsoft to the European Union was that the merger
strengthened the company overall in order to provide a viable computing store on iOS and Android.
</item 22>

<item 23>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Informatica

Reported on January 11 that it will lay off 7% of it’s workforce, or 450 staffers globally.

Carta

Announced on January 11, the equity management platform cut 10% of its staff. Judging by LinkedIn data, the layoff could have impacted around 200 employees.

Citizen

Impacting 33 staff members on January 11.

Coinbase

To cut 950 jobs, or about 20% of its workforce, and shut down “several” projects, announced on January 10. This is the second round of major layoffs at the
crypto exchange, which eliminated 18% of its workforce, or nearly 1,100 jobs last June.

SuperRare

The NFT marketplace is cutting 30% of its staff, announced on January 6.

Amazon

Announced on January 5, eliminating more than 18,000 roles. This announcement extends a previously announced round of layoffs in November of 2022. On January
19, the company announced it would end AmazonSmile.
</item 23>

<item 24>
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
</item 24>

<item 25>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
We established a market worldwide, excluding China. So any remedies, we would presume, would be worldwide. We also have the European DMA; we have Epic v. Apple
and Epic v. Google cases coming up in Australia, and another one in the UK.

It’s not just Epic anymore, there’s a lot of legislators, there’s a lot of regulators, and there’s other litigation all pushing in the direction of openness.
And we’re going to do absolutely everything we can.

Do you have anything to say to your former partners in litigation, Match Group, that abandoned you at the last minute and may now be regretting it?

Oh, yeah, no. Match has been an awesome partner and a fellow member of the Coalition for App Fairness. I really hope that they got out of their settlement what
they needed to get for their business. Very few companies have the resources that Epic has to fight multinational litigation against the world’s two most
powerful companies.
</item 25>

<item 26>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
The gaming industry, which likes to boast about how much money it makes in comparison to other forms of entertainment, is laying off workers nearly every week
at top studios. Meanwhile, the overall gaming market is shrinking. The cracks in the system that produces and promotes big-budget games are getting hard to
ignore.

“I wouldn’t trade it for the world,” Tenney said in his goodbye video, which is overflowing with footage of esports casters and viewers in awe of his Fortnite
plays.

“I’ve been doing this since I was a kid, man,” he continued, shaking his head. “I feel like my childhood and even some of my adulthood kind of flushed away. I
feel like this is the end of the journey, man.”
</item 26>

<item 27>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Lyft

Announced on April 21 that employees will learn whether they have a job or not via an email that will be sent out April 27.

Lyft layoffs to affect 26% of workforce, or about 1,072 people as promised on April 27.

Meta

Announced April 18th that it is expected to lay off 10,000 jobs in the coming months. This is on top of the 11,000 jobs that were cut in November.

Redfin

Announced April 13th that it has laid off 201 employees, about 4% of its workforce. This is the third time the Seattle-based real estate company has reduced its
workforce since June.

Apple

Bloomberg reported on April 3 that Apple is laying off a small number of roles on its corporate retail teams.

March 2023

Netflix

Announced on March 31, Netflix confirms a “handful of layoffs,” which includes two longtime executives. The exact number of layoffs is unclear.
</item 27>

<item 28>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
The most extreme negative outcomes for an acquired indie studio are, of course, layoffs and closures. We saw a lot of these in 2023.

Layoffs

The post-acquisition power dynamic is playing out in public and in real-time. It’s estimated that more than 9,000 people in video games were laid off this year
and the firings affected teams of all sizes. This is a crisis amount of cuts. In 2022, just 1,000 video game jobs were lost, according to layoffstracker.com.

The Embracer Group provides the clearest example of rampant, surprise layoffs in 2023. Embracer has spent the past few years acquiring prominent midsize
studios, including Gearbox Software (Borderlands), Crystal Dynamics (Tomb Raider), Eidos-Montreal (Deus Ex) and Square Enix Montreal (Deus Ex Go).
</item 28>

<item 29>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So this trial has been four years in the making. I look back at a September 2019 email that was in discovery about a plan to draw Google into a legal battle
over antitrust. Can you describe what it felt like, after four years, to hear the jury find for Epic?

Well, it’s a great day for all developers to see that the Sherman Antitrust Act works in the new era of tech monopolies; we’ve not had a major antitrust verdict
against a tech company that meant change and benefits for everybody since the 1990s, with the US v. Microsoft. Back in the early days of the internet. So this
is an awesome thing and it’s much needed by the industry which is being strangled by a few gatekeepers imposing insane amounts of control and extracting huge
taxes, which not only raise prices for consumers but also make a lot of kinds of products just unviable.

I understand the significance, but... you were there in person.
</item 29>

<item 30>
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
</item 30>

<item 31>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Original story continues:

Llamacorn what now? What is happening?

Before we get to the Llamacorn, a little background. Epic is of course the studio behind Fortnite, the extraordinarily popular free-to-play game. Fortnite makes
money by selling in-game items with its virtual currency V-Bucks. Players often buy V-Bucks the same place they play Fortnite. And until August 13th, 2020, if
the player used an Android or iOS device and installed the game through an official app store, that purchase triggered an in-app payment fee to Google or Apple.

Critics call such fees the “Google tax” or the “Apple tax,” and Epic definitely wasn’t a fan.

When Epic decided to take action against these respective “taxes,” it made August 13th, 2020 a very busy day for Apple, Google, Epic, and us here at The Verge.
First, Epic announced it was bypassing Apple and Google’s app store fees.
</item 31>

<item 32>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
No other teams are impacted.

Divvy Homes

Announced on September 11 that it is laying off 94 employees.

Sensor Tower

Announced on September 8 that it laid off approximately 40 people out of the 270+ at the company, including C-suite executives.

Roku

Announced September 6 that it plans to lay off approximately 10% of its workforce, representing over 300 employees.

August 2023

Malwarebytes

Announced on August 31 that it laid off 100 employees, mainly impacting corporate employees.

Twiga

Announced on August 21 that it is laying off 283 employees, or 33% of its workforce.

SecureWorks

Announced on August 14 that it will let go of 15% of its workforce, impacting about 300 employees. This is the company’s second round of layoffs this year.

CoinDesk

Announced on August 14 that it is cutting 16% of staff.
</item 32>

<item 33>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Didn’t other parties sue Google too?

They did! And then, they all settled. All 50 state attorneys general and a number of consumers have tentatively settled with Google over app store antitrust
claims, though it’s not clear what the deal might include.

Match Group, the company behind Tinder, Match.com, OkCupid and other dating apps, just reached a surprise last-minute settlement too, one that makes it sound
like Match effectively gave up.

Epic stands alone.

Could we get a settlement between Google and Epic as well?

Epic seems to have no interest in settling as long as the “Google tax” exists, and the company’s repeatedly said it’s not interested in cash. So far, the
company’s been willing to push these cases all the way through multiple appeals, cost be damned.

It’s not clear what Google could proactively offer that would satisfy Epic, since CEO Tim Sweeney has also publicly suggested that Google would simply
circumvent any settlement it offers.
</item 33>

<item 34>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Or at least it was worse, before they were destroyed.”

The best Google could do in response was to plant its own feeble seed with the jury, too: “Is Epic using the chats to distract me from all the evidence I do
see?”

“It’s true that Google could have automatically saved all chats for all relevant employees, but just because Google didn’t save some chats didn’t mean it
violated antitrust laws,” Pomerantz argued.

Epic’s opening statements seemed to paint a clearer picture for the jury than those from Google. But things got complicated for both parties when the first two
witnesses — Epic Games Store head Steve Allison and Yoga Buddhi CEO Benjamin Simon, who also appeared in the earlier Epic v. Apple trial — took the stand.

Both Epic and Google spent a long, long time on subtle lines of questioning.
</item 34>

<item 35>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Crunchbase

Announced on LinkedIn on July 9 that the company is laying off 52 employees.

Microsoft

Announced July 10 that the company is eliminating additional jobs a week after the start of its 2023 fiscal year. These layoffs are in addition to the 10,000
layoffs announced in January.

ClickUp

Announced on July 4 that it has laid off 10% of its workforce, about 90 people.

June 2023

Niantic

Announced June 29 that it laid of 230 employees — one year after it laid off around 90 employees.

Plex

Announced June 29 that it laid off approximately 20% of its staff, or 37 employees.

Grab

Announced the week of June 23 that it laid off more than a thousand people, or 11% of its staff.

Olx Group

Announced on June 20 that it has cut around 800 jobs globally. The news comes as the company started to close operations of its automotive business nit Olx
Autos in some markets.
</item 35>

<item 36>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
The lawyers can see them, but I saw almost all the key elements in this case at the same time the jury saw them. Two things stand out big time. Google’s Project
Hug was an astonishingly corrupt effort at a massive scale — the same corrupt deal structure they tried to do with Epic when they wanted to pay us off to launch
Fortnite on Google Play and not launch our own store.

As soon as we thwarted their effort, they went around to 27 different developers and offered each one a payoff to undermine any effort we had to get their games
onto our store exclusively. Activision and Riot and Supercell had direct distribution plans that they were planning on; Google paid them not to pursue those
plans. Just direct blatant violations of anti-competition law, it’s crazy a company of Google’s scale would do that.

“If you’re a smaller developer than Spotify, you get screwed.”

The other astonishing one was the Spotify deal. Spotify is the one company that had comparable negotiating power to Fortnite.
</item 36>

<item 37>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
100 Thieves

Is reducing its workforce by 20%, CEO John Robinson announced November 2. The esports brand is spinning out its game studio as well as its energy drink brand,
Juvee.

OpenSpace

Conducted a round of layoffs, CEO Jeevan Kalanithi announced November 2. The number of employees impacted by the layoffs has yet to be reported.

Informatica

Is laying off 10% of its global workforce, the cloud data management company announced in its Q3 results November 1.

Splunk

Is set to cut around 500 employees CEO Gary Steele said in a message to employees November 1 ahead of its planned acquisition by Cisco.

October 2023

Bungie

Is laying off about 100 employees, CEO Pete Parsons announced 10/30. The Sony-owned game studio is also delaying two of its upcoming titles as a result.

Karat

Reportedly cut an estimated 10% of its workforce October 27 after the financial startup for creators raised a $70 million Series B round earlier this year.
</item 37>

<item 38>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Multiverse

Will lay off nearly a third of their U.S. staff. It’s the second round of layoffs at the U.K.-based unicorn company in just over a month.

ONE

Is cutting its workforce by around 25% November 27, citing “market conditions” affecting the EV battery startup.

ByteDance

Started mass layoffs November 27 in its gaming department, Nuverse, after two years of tepid performance. The number of affected employees is unknown.

Veev

Is closing up shop, according to multiple reports, after reaching unicorn status in 2022.

Tulip

Laid off 25 employees, CEO Ali Asaria announced November 24.

Anar

Is shutting down operations and returning capital to investors, CEO Nishank Jain announced November 23.

Alerzo

Cut 100 workers November 21, reportedly due to increases in automation. The recent layoffs come after Alzero cut 15% of its workforce earlier this year and 5%
in 2022.
</item 38>

<item 39>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
A GitHub spokesperson told TechCrunch that the layoff is part of the streamlining effort the firm had disclosed in February to cut roughly 10% of its workforce
by end of Q1 2023.

Disney

On March 27 in an internal memo to employees, Disney revealed there will be three rounds of layoffs, the first beginning this week. The job cuts will impact
approximately 7,000 employees, which was announced in February.

Salesforce

On March 24, Bloomberg reported that more layoffs could be on the way at Salesforce, quoting chief operating officer Brian Millham, who indicated that the
company could be adding to the ongoing job cutting at the CRM leader and in tech in general. If the layoffs happen, it would come on top of the 10% cut in
January.

Accenture

Announced March 23 that it plans to cut 19,000 jobs, or 2.5% of its workforce.

Indeed

Announced March 22 that it will lay off 2,200 employees, or 15% of its staff.
</item 39>

<item 40>
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
</item 40>

<item 41>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 41>

<item 42>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Snap

Laid off fewer than 20 project managers November 8. CNBC reported that Snap’s vice president of engineering, ​​Nima Khajehnouri, will also be leaving.

Amazon

Started cutting workers in its music streaming division in Latin America, North America and Europe, the company confirmed November 8.

Google

Confirmed November 8 it eliminated “a small number of roles” in its consumer support staff, Google Users & Products.

Virgin Galactic

Said in a memo to staff November 8 it is cutting costs and reducing staff. The number of employees impacted is currently unknown.

Ubisoft Montreal

Confirmed November 7 it will cut 98 roles in business administration and IT. Ubisoft said 124 positions will be eliminated overall across Canada, including
reductions in its global IT team and its VFX studio, Hybride.

Nextdoor

Revealed in its Q3 earnings that it is reducing its workforce by 25%.

Beyond Meat

Announced cuts November 6 affecting around 19% of its non-production employees.
</item 42>

<item 43>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
So there are absolutely no hard feelings, and we’re grateful that they joined the case because they did help in critical ways. Epic will continue to fight for
all developers, seeking remedies, and... I don’t know if Sundar is going to be calling me, but if he does, all of our discussions will be directed toward
solving the problem for everyone.

A lot has changed since you filed the original lawsuit over three years ago. For a bit there, it seemed like Epic had an interest in expanding outside of games
with acquisitions like Houseparty and Bandcamp. What changed? And did app store restrictions play into that at all?

Let’s see. No... well, not directly, anyway. We have a really broad strategy of building games and technology. We’ve had huge success in recent years with the
Unreal Engine gaining adoption among all kinds of industries even beyond games. We’ve also run into our own financial limitations. You know, we expanded the
company to nine times the size we were before Fortnite took off.
</item 43>

<item 44>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
February 2023

Twitter

Announced on February 26, the company laid off more than 200 employees, including Esther Crawford, Haraldur Thorleifsson and Leah Culver. Since Musk took over
Twitter in October last year, the company’s headcount has fallen by more than 70%.

Poshmark

Announced February 24, Poshmark confirmed with TechCrunch that less than 2% of its workforce was affected, primarily in the U.S. The company employs roughly
800+ employees.

Green Labs

We do not have an exact figure of how many Green Labs plans to lay off its staff. Green Labs confirmed to TechCrunch that it is conducting a round of layoffs
that could impact at least 50% of its workforce.

Chipper Cash

Announced on February 20, the African cross-border payments platform conducted a second round of layoffs just 10 weeks after it cut approximately 12.5% of its
workforce. Chipper Cash relieved almost one-third of its workforce, about 100 employees.
</item 44>

<item 45>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 45>

<item 46>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Insightec

Is reportedly laying off 100 employees, with 60 from the company’s Israel headquarters.

Udaan

Reportedly laid off up to 150 employees in a restructuring effort. This month, the business-to-business e-commerce startup scored $340 million in new funding.

Enphase

Is reducing their workforce by around 10%, impacting roughly 350 contractors and employees, president and CEO Badri Kothandaraman wrote in a blog post December
18.

Arm Holdings PLC

Reportedly laid off more than 70 software engineers in China and will relocate some roles.

Delivery Hero

Is shutting down operations in Taiwan and Turkey and plans to eliminate roles at its Berlin headquarters, the company said December 18.

eBay

Is reportedly eliminating nearly 10% of its workforce in Israel. It’s the company’s second round of layoffs this year.

Intel

Will cut 235 employees at its Folsom office campus at the end of the year, the company reported to state employment officials December 15.
</item 46>

<item 47>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Alphabet

Google’s parent company announced laying off 6% of its global workforce on January 21, equating to 12,000 employees. These cuts impact divisions such as Area
120, the Google in-house incubator and Alphabet’s robotics division, Intrinsic.

Fandom

The entertainment company announced an unspecified number of employees impacted across multiple properties on January 20. According to a report by Variety, the
company employs around 500 people, and the layoffs have affected roughly 10% of its staff across different sites.

Swiggy

Announced plans to lay off 380 jobs on January 20 and shut down its meat marketplace.

Sophos

Announced on January 18, 10% of its global workforce, about 450 people were let go.

Microsoft

As announced on January 18, 10,000 employees will be impacted.

GoMechanic

Laid off 70% of its workforce on January 18.

Clearco

Announced on January 17, impacting 30% of staff across all teams.
</item 47>

<item 48>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
Microsoft

As a part of its recent announcement to layoff 10,000 people, Microsoft laid off an entire team dedicated to guiding AI innovation that leads to ethical,
responsible and sustainable outcomes. On March 27, Microsoft laid off 559 workers from its Bellevue and Redmond operations.

Meta

CEO Mark Zuckerberg confirmed rumors March 14 that the company will be cutting 10,000 people from its workforce and around 5,000 open roles that it had yet to
fill.

Y Combinator

Announced March 13 that it will impact 20% of staff, or 17 team members.

Salesforce…continued

Salesforce first announced that it was laying off 10% of the workforce in January, but some employees didn’t know until February. The week of March 10, more
employees are just learning they have been laid off. Salesforce confirmed that these layoffs were part of the 10%.

Atlassian

Announced on March 6, Atlassian is laying off about 500 employees, or 5% of its total workforce.
</item 48>

<item 49>
A comprehensive list of 2023 tech layoffs (TechCrunch, 2023-09-28T20:04:20+00:00)
CoinDesk

Announced on August 14 that it is cutting 16% of staff.

NCC Group

Confirmed on August 10 that it is making more layoffs, its second round of layoffs in just six months. The company declined to say how many employees are
affected by this round of cuts. NCC Group cut 125 employees in February.

Rapid7

Announced on August 9 that it plans to lay off 18% of its workforce, affecting more than 400 employees globally.

Mobile Premier League

Announced on August 8 that it is laying off approximately 50% of its employees or 350 people.

Astra

Announced on August 4 that it has laid off 25% of its workforce since the beginning of the quarter.

Discord

Announced on August 3 that it has let go of nearly 40 employees or 4% of its workforce.

HackerOne

Announced on August 2 that it is laying off 12% of its workforce, approximately 50 employees.
</item 49>

<item 50>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
Both Epic and Google spent a long, long time on subtle lines of questioning. You really had to read between the lines to see that Epic was trying to make a
point about how Google’s 70/30 revenue split is probably based on an arbitrary decision Valve made two decades ago with Steam or how Google was trying to make a
point that Epic, too, likely believed that an app store provides more value than just payment processing and maybe deserves more money.
</item 50>
=== END ITEM id=9a4fbb4b5b2bbbd8 ===

=== ITEM id=62799b4e989c57d1 ===
Gold sentences and facts:
Sentence 0: Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.
  Fact 0.0: Zazie Beetz will return in the role of Arthur's neighbor.
  Fact 0.1: Zazie Beetz's character Sophie is Arthur's love interest.
  Fact 0.2: Sophie is unwitting about being Arthur's love interest.
Sentence 1: But then, there’s MJ as well who has made a comeback from the first installment.
  Fact 1.0: MJ is also in the new installment.
  Fact 1.1: MJ has made a comeback from the first installment.

Retrieved items:
<item 1>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
(It’s up.) Figuring out how to get there is another matter.

Even in games that prominently feature it, climbing is often functional, at best — whether it’s Link refusing to acknowledge handholds are a thing or Nathan
Drake snapping to ledges with a level of magnetism only thought possible at CERN.

Jusant is the first and only game I’ve played that gets it — that sees the inherent grace in the sport and portrays it on the screen, not as a means of getting
somewhere, but as the reason to go there in the first place. —Ari Notis

8. Marvel’s Spider-Man 2

Developer: Insomniac Games

Where to play: PlayStation 5

Marvel’s Spider-Man 2 builds on the successes of the first game, both narratively and in its immensely satisfying gameplay. The game’s story takes a familiar
Spider-Man narrative — that of friends turned foes — and adds new twists, building out compelling villains (and demanding ones) in the process.
</item 1>

<item 2>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Maybe Hollywood as a whole took away the wrong lesson (we’ve seen so many multiverse movies), but Into the Spider-Verse redefined genre staples.

The sequel continues to raise the bar. It’s a little busier than the first, pulling in more characters, more frantic visuals, more emotional throughlines, but
it’s still bold, daring, and unlike anything else in the superhero movie sphere.

As the characters bounce across the multiverse, the animation style shifts, with each character rooted in a different style. Spider-Gwen’s world is rendered
with watercolors, which shift according to the tone of her story. She gets a step up in this movie — a second chance for Gwen Stacy, whose previous iterations
across media have not ended well. Meanwhile, Spider-Punk, the brash counterculture rebel, is made up of collaged bits, and animated against the norm so he
stands out even more. Every frame of the movie is a visual feast, just bursting with lovingly rendered details.
</item 2>

<item 3>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.
</item 3>

<item 4>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
But director and co-writer Matt Reeves found an effective take on the Dark Knight that resonated with audiences and landed solidly at the box office, and DC
immediately greenlit a sequel, again with Robert Pattinson as the Caped Crusader. No plot details are clear yet, though Pattinson and Reeves have been
blue-skying about a lot of things they’d like to do: The Court of Owls storyline, maybe? Mr. Freeze? Calendar Man, for god’s sake?

Undated

The Penguin

HBO Max’s TV spinoff of Matt Reeves’ The Batman is meant to bridge the gap between that movie and Reeves’ sequel, as Penguin (Colin Farrell) solidifies power in
the Gotham underworld immediately after the death of Carmine Falcone (John Turturro) in the movie.
</item 4>

<item 5>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
One of the Miles-specific side quests involves learning about the artistic history of Harlem, and it entwines the real-life histories of Josephine Baker,
Langston Hughes, and Charlie Parker with the fictional history of SHIELD and other Marvel properties. An early mission shows off Insomniac’s technical chops
with a section that cleverly borrows a trick from Ratchet & Clank: Rift Apart. Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to
break free of expectation and canon.

[Ed. note: Spoilers follow for Spider-Man: Across the Spider-Verse.]

The recent Spider-Man: Across the Spider-Verse has an obsession with “canon events,” pivotal moments in a Spider-Man’s life that the film’s villain, Spider-Man
2099, is dedicated to maintaining across the multiverse.
</item 5>

<item 6>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.
</item 6>

<item 7>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2023

Aquaman and The Lost Kingdom: Dec. 20

Expect online fandom debates to go ballistic again as we get closer to the sequel to James Wan’s joyously goofy 2018 under-the-sea extravaganza, Aquaman. Jason
Momoa is back as the titular superhero/sea king, with Nicole Kidman returning as his mother, Atlanna. Ben Affleck is reprising his role as Bruce Wayne/Batman,
Patrick Wilson is back as undersea usurper Orm the Ocean Master, Yahya Abdul-Mateen II is back as the villain Black Manta, and Wan is back at the helm. But most
of the early noise about Aquaman and the Lost Kingdom has been about the return of Amber Heard as Aquaman’s love interest, Mera — after the long, contentious
legal battle between Heard and Johnny Depp, expect the lead-up conversation about this one to be noisy.

2024

Joker: Folie à Deux: Oct.
</item 7>

<item 8>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
This will be the first appearance of WildStorm canon in live-action film.

Supergirl: Woman of Tomorrow

Just as Superman: Legacy is meant to present a new view of Superman for the DCU, Woman of Tomorrow is being billed as “a science fiction adventure” that
“presents a Supergirl viewers are not used to seeing.” The movie will be based on Tom King’s celebrated Supergirl comics, and will presumably not touch on
anything from the six-year run of Supergirl on The CW starring ​​Melissa Benoist.

Swamp Thing

The recent appearance of Man-Thing in Marvel’s black-and-white special Werewolf by Night suggested there can be space for mainstream superhero comics’
occasional forays into EC Comics-style “weird horror” territory. But it’ll be a while before we get to see what that looks like on the DC side.
</item 8>

<item 9>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.
</item 9>

<item 10>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
If you want curated lists of our favorite media, check out What to Play and What to Watch .

Retellings are compelling because they allow writers to toy with readers’ preconceived notions of how the story “should go,” both in terms of genre conventions
and in terms of plot. We know Little Red Riding Hood will go to her grandmother’s house, where she will find that her grandmother has been eaten by a wolf, just
as we know Spider-Man will be bitten by a radioactive spider. The fun, both for the author and the reader, comes in deviating from those norms — in finding new
ways to tell old tales.

With Marvel’s Spider-Man 2, Insomniac Games attempts something similar.

Mainstream Spider-Man stories are increasingly Spider-Men stories, and Spider-Man 2, the third Spidey game from Insomniac, is no different.
</item 10>

<item 11>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.
</item 11>

<item 12>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
You play as both the yassified Peter Parker from Spider-Man Remastered and Miles Morales from his eponymous debut in 2020, switching between the two at your
leisure with all the speed afforded by the PlayStation 5’s SSD.

In fact, speed is one of the primary improvements on display in the first PS5-exclusive Spider-Man game. New to the series are web wings, which allow the
Spider-Mans to fly through the city, gliding over rooftops and gaining speed from wind tunnels without needing to rely on buildings, bridges, or other anchors
for web-slinging. (They’re especially useful when crossing the East River into the newly added Brooklyn and Queens.) Web-swinging is faster now as well,
enhancing that ineffable feeling of being Spider-Man as you glide through New York with ease. Whether you’re playing as Miles or Peter, traversing the three
available boroughs is an undeniable joy — so much so that I found myself using fast travel less frequently than in the previous two games, because movement felt
that much more fluid.
</item 12>

<item 13>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct. 20 on PlayStation 5. The game was reviewed using a pre-release download code provided by Sony Interactive
Entertainment. Vox Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via
affiliate links. You can find additional information about Polygon’s ethics policy here .
</item 13>

<item 14>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Whether it’s the upside-down kiss, the death of Uncle Ben or Jefferson Davis, or simply the spider bite that starts it all, Spider-Man 2099 believes that
upholding these canon events, even the painful ones, is key to preserving the stability of each strand of reality. Miles disagrees, believing it is Spider-Man’s
job to resist the seemingly inevitable — to always do what’s right, not simply what’s preordained. The film ends with a promise of pushing back against
Spider-Man 2099, and with it, the established canon.

I was hungry for Spider-Man 2 to push back in a similar way — to treat the established story like Angela Carter repurposing “Little Red Riding Hood” to her
whims, like Michael Cunningham plumbing “Rumpelstiltskin” for a story about the desire to have a child, or like Helen Oyeyemi and her body of work.
</item 14>

<item 15>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
The DC Extended Universe, as DC Comics’ movie and TV line was originally called, has gone through some major growing pains since it launched with Zack Snyder’s
Man of Steel in 2013. Looking for ways to compete with Marvel Comics’ massively popular Marvel Cinematic Universe, DC has repeatedly switched tracks and tactics
on the DCEU, sometimes steering toward a single interconnected universe, then backtracking into a more director-driven plan where movies like Todd Phillips’
Joker could be made without fitting into any specific previous continuity.

In 2023, the Snyderverse is coming to an end, slowly rolling to a halt with movies like Shazam! Fury of the Gods, The Flash, and Blue Beetle, all leading up to
the grand finale of Aquaman and the Lost Kingdom. Once the last bits of the DCEU make their way to theaters, it will be time to usher in a new era of DC movies.
</item 15>

<item 16>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?
</item 16>

<item 17>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2024

Joker: Folie à Deux: Oct. 4

Todd Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with ​​Joaquin Phoenix reprising his role as
Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest,
Sophie.
</item 17>

<item 18>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?
</item 18>

<item 19>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.
</item 19>

<item 20>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Waller

Gunn’s hilarious, heartfelt Suicide Squad spinoff series Peacemaker was greenlit for a second season in February 2022, but Gunn has now said he’s going to have
to put that project on hold for a bit to concentrate on other things — including Waller, the previously untitled Peacemaker spinoff centered on Viola Davis as
no-nonsense supervillain resource broker Amanda Waller. The character, introduced to DC’s movie continuity in David Ayer’s little-loved 2016 version of Suicide
Squad (though she’s been part of the comics continuity since 1986), is one of the few elements of Ayer’s movie that was held over in Gunn’s 2021
reboot/reshuffle movie, called The Suicide Squad. She also turned up for a quick cameo in Black Adam. Now she’s getting her own show — which “features Team
Peacemaker,” so it may feel like a stealth season 2, depending on who from the team winds up on the series.
</item 20>

<item 21>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
“Those slavering jaws; the lolling tongue; the rime of saliva on the grizzled chops.”

This is Angela Carter, not on Venom, but on the wolf from “Little Red Riding Hood” in The Bloody Chamber, her collection of reimagined folk and fairy tales.
It’s actually from “The Company of Wolves,” which is one of three retellings of “Little Red Riding Hood” in the collection, each with its own spin on the
familiar tale.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences. When we award the
Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth fitting into your
schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .
</item 21>

<item 22>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 22>

<item 23>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to break free of expectation and canon

Narratively, juggling two Spider-Mans requires a different kind of deftness. Two villains new to the series take center stage — Kraven the Hunter and Venom — as
well as a returning cast of nearly every villain from the previous two games. Without delving into spoilers, a large majority of that cast is sidelined early
on, leaving Kraven and Venom as our principal antagonists, albeit with large roles for returning villains the Lizard (who made a brief appearance in Miles
Morales) and Mister Negative. There are others besides them that should delight comic fans, setting up future conflicts for either expansions or the presumed
Spider-Man 3 to tackle. If what you come to Spider-Man for is spectacle and villainy, Spider-Man 2 has the goods and then some.

As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up.
</item 23>

<item 24>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
They also turned up in season 3 of Max’s Doom Patrol series. The initial press release on Dead Boy Detectives said it would have an eight-episode first season,
described as “a fresh take on a ghost story that explores loss, grief, and death through the lens of Edwin Payne (George Rexstrew) and Charles Rowland (Jayden
Revri), two dead British teenagers, and their very alive friend, Crystal Palace (Kassius Nelson).”

Peacemaker season 2

Alongside everything else in this universe with his name on it, Gunn hasn’t forgotten Peacemaker. According to the director, Christopher Smith will be making
his return for season 2, sometime after Gunn wraps production on Superman: Legacy. This likely means that season 2 won’t be coming along until sometime in 2025
or even later, but hey, at least we know it’s out there.
</item 24>

<item 25>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
There’s a visible growth in her arc especially considering her profession as a journalist. Yes, she’s Peter’s love interest, but that’s not everything she’s
there for. Players can easily see it happening throughout the game’s duration.

The verdict?

ADVERTISEMENT Article continues below this ad

It’s obvious to expect that a lot of fans who didn’t like MJ’s missions in the first installment won’t like her this time as well. But considering how she was
before and how she’s in Spider-Man 2, the overall change is quite praiseworthy. Things are different this time, thankfully.

MJ’s missions are now having some actual meaning attached to them rather than acting like fillers. Players who want to have a few interesting moments aside from
the main quests featuring Miles and Peter can have a sense of finishing well-designed stealth quests. Mary’s gameplay missions end up increasing the overall
quality of Spider-Man 2.
</item 25>

<item 26>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up. Their two stories are threaded together, allowing for randomized
encounters where Peter might show up while you’re playing as Miles (or vice versa), with the two of you tag-teaming a random criminal and slamming them into the
ground in what feels like the world’s least fair fight. And yes, they do the Spider-Man-pointing-at-Spider-Man meme in-game. And yes, it is charming the first
time (though maybe less so the second). The main story sometimes swaps Spider-Mans mid-mission, as in the dramatic opening sequence where Sandman coats the city
in dust as Peter and Miles rush to help firefighters and first responders, with eerie echoes of 9/11. Side missions often allow either Spider-Man to tackle
them, but the main story beats require one or the other, as each hero is allowed his full particularity as a singular Spider-Man.
</item 26>

<item 27>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Peter’s plot involves a retelling of the Venom origin story, with beats that will be recognizable to those even passingly familiar with the Spider-Man mythos.
If Spider-Man 2 is a game of retellings, Peter’s story clings most closely to the canon. If you’ve seen Sam Raimi’s Spider-Man 3, you’ll know the gist of what
to expect when the symbiote finds its way to Peter. It isn’t until the late game that Peter’s story starts pushing more strongly against our canonical memories
and into something more unique to Insomniac.

Miles’ story takes more liberties. Hailey Cooper, Miles’ love interest from his solo game, returns here, taking more of an MJ-esque role and carving out her own
identity separate from Miles’. American Sign Language is featured heavily throughout Miles’ missions because of Hailey’s presence, feeling as natural and
integral to the story as anything from the canon.
</item 27>

<item 28>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Guardians of the Galaxy and Suicide Squad director James Gunn and producer Peter Safran are taking the reins of the newly branded DC Studios, and they’re
bringing a long-term plan to DC with them. The new projects on the DC docket are meant to reboot or reshape DC’s best-known heroes, while reaching into some
deeper corners of the DC canon for more obscure characters. And it’s wrapping up the DCEU era, replacing it with just “the DCU,” a mainline continuity that will
exist alongside other DC film and TV universes. Here’s everything DC is currently planning to put in theaters and on TV screens, in 2023 and well into the
future.

[Ed. note: This story only covers Warner Bros.’ DC projects and doesn’t include The CW’s Arrowverse, which is effectively ending in 2023.]

2023

Aquaman and The Lost Kingdom: Dec.
</item 28>

<item 29>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Or, to give a more mass media example, like Star Wars: Visions, which gives creative leeway to a variety of animators and storytellers to play with the very
basics of what makes Star Wars Star Wars.

Spider-Man 2 doesn’t go quite this far, but where it does go is still satisfyingly slant. Wrinkles to critical characters and plot points will keep even
seasoned Spider-fans on their toes. Moreover, it plays like a dream, with smart additions to combat that enhance the power fantasy of being Spider-Man. Each
Spider-Man now has a full arsenal of special moves and gadgets that complement one another, making even the most basic combat encounter feel like a
choreographed ballet of kicks, thwips, and zaps. The game is beautifully rendered, taking full advantage of the newest PlayStation, with such detailed textures
on each unlockable suit that you’d swear they were real, and ray-traced window after ray-traced window in a loving rendition of New York.
</item 29>

<item 30>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.

Creature Commandos

James Gunn is listed as the writer for this seven-episode adult animated series, which will also include Suicide Squad head Amanda Waller. This time, she
“creates a black ops team out of monstrous prisoners.” The team will include Frankenstein’s monster (here called Eric Frankenstein and voiced by David Harbour),
who has a long, complicated DC history of his own. The series is set to star Frank Grillo (Prison Break) as Rick Flag Sr., Maria Bakalova (Guardians of the
Galaxy Vol. 3) as Princess Ilana Rostovic, Indira Varma (Game of Thrones) as the Bride of Frankenstein, and Alan Tudyk (Harley Quinn) as Doctor Phosphorus.
</item 30>
=== END ITEM id=62799b4e989c57d1 ===

=== ITEM id=1b42ae5c4ed9bede ===
Gold sentences and facts:
Sentence 0: It will attempt to argue that Google restrains competition within two separate markets, including the distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.
  Fact 0.0: The planned argument will attempt to argue that Google restrains competition within two separate markets.
  Fact 0.1: The two markets include the distribution of apps to Android users.
  Fact 0.2: The two markets include the market for payment processing solutions for content inside Android apps.
Sentence 1: Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part.
  Fact 1.0: Wired removed the article for not meeting its standards.
  Fact 1.1: The degree to which the article was shared on social media shows just how willing people are to accept foul play on Google's part.
  Fact 1.2: The article was boosted in write-ups at other outlets.
  Fact 1.3: The degree to which the article was boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google's part.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
What to know about Fortnite maker Epic Games’ antitrust battle with Google, starting today (TechCrunch, 2023-11-06T16:47:11+00:00)
To make its case, Epic plans to bring claims under Sections 1 and 2 of U.S. antitrust law, the Sherman Act, as well as California’s anticompetitive laws, the
Cartwright Act, and the Unfair Competition Law. It will attempt to argue that Google restrains competition within two separate markets, including the
distribution of apps to Android users and the market for payment processing solutions for content inside Android apps.

Google argues its commissions aren’t just tied to billing, but offers discount for third-party billing options

Today, Google requires apps to use its own first-party billing system, and charges app developers a 15% to 30% commission on the sales it processes.

However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has
declined to use.
</item 1>

<item 2>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
The DOJ has said that Google gets 16 times more data than Bing does everyday.

The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to
illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether
it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.

Enforcers will want to show that antitrust law is still relevant and can successfully take on the biggest, most powerful companies in the world.
</item 2>

<item 3>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
“This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation
where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users
changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one
search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging
all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact
that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google?
</item 3>

<item 4>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A new class action lawsuit filed this week in the U.S. District Court in D.C. accuses Google and parent company Alphabet of anticompetitive behavior in
violation of U.S. antitrust law, the Sherman Act, and others, on behalf of news publishers. The case, filed by Arkansas-based publisher Helena World Chronicle,
argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means. It also specifically cites new AI
technologies like Google’s Search Generative Experience (SGE) and Bard AI chatbot as worsening the problem.

In the complaint, Helena World Chronicle, which owns and publishes two weekly newspapers in Arkansas, argues that Google is “starving the free press” by sharing
publishers’ content on Google, losing them “billions of dollars.”

In addition to new AI technologies, the suit points to Google’s older question-and-answer technologies, like the “Knowledge Graph” launched in May 2012, as part
of the problem.
</item 4>

<item 5>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
On October 19, Expedia’s former chief operating officer, Jeff Hurst, told the court the company’s ad fees increased tenfold from $21 million in 2015 to $290
million in 2019. And yet, Expedia’s traffic from Google did not increase. The implication was that this was due to direct competition from Google itself. Hurst
pointed out that Google began sharing its own flight and hotel data in search results in that period, according to The Seattle Times.

European antitrust fine spurs Google to beef up search engine quality

The government argued on November 10 that Google only endeavored to enhance its search engine in the European Union after it was hit by a €5 billion antitrust
fine in 2018, internal documents revealed, according to Bloomberg.

The EU’s antitrust order forced Google to offer Android phone users a screen with five search engine options to choose from, according to the DOJ.
</item 5>

<item 6>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai,
according to The Verge.

Google understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy
meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their
Google searches dropped 27%.

“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that
summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.

The amount that Google spent on homepage market share has been a fixing point in the trial.
</item 6>

<item 7>
Tim Sweeney on Epic’s victory royale over Google (The Verge, 2023-12-12T19:17:14+00:00)
What would you say the differences are between the Apple and Google cases?

I would say Apple was ice and Google was fire.

The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the
same terms, they force OEMs and carriers to all have the same terms.

Whereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re
paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and
it’s right there for everybody to read and to see plainly.

I think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing,
whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute.
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
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.

Rosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or
Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.

Google also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome
web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other
companies to make or sell more Android devices, he said.
</item 9>

<item 10>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the
piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which
it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of
privacy invasion and inappropriate data handling follows the company at all times.

Grey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may
have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s
official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not
sponsored.
</item 10>

<item 11>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo

We’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took
Microsoft to trial in the 1990s — and the revelations just keep getting juicier.

In our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have
Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s
most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.

Before we go into some of these tidbits…

Why the Google vs. U.S.
</item 11>

<item 12>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down
to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.

For the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us,
it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has
been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired
posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates
people’s search queries for maximum ad revenue.
</item 12>

<item 13>
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
</item 13>

<item 14>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three
Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.

Meanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search
— that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was
making a big push to win business with people who wanted to buy identity theft protection.

Revenue-shares to pre-install Google apps on Androids

Jamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between
Google and Apple is “as intense as it gets,” reports Bloomberg.
</item 14>

<item 15>
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
</item 15>

<item 16>
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
</item 16>

<item 17>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
A Google spokesperson offered a statement on the lawsuit, saying “This lawsuit is meritless. People have many ways to access information and news content today
– through publishers’ websites, dedicated apps, social media platforms, print papers and more. Google links people to publishers’ websites more than 24 billion
times each month – at no cost to them.”

The complaint is available below.

Helena World Chronicle, LLC v. Google LLC and Alphabet Inc by TechCrunch on Scribd

Editor’s note: This post was updated after publication with a statement provided by Google on Friday evening.
</item 17>

<item 18>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
In addition to damages, the suit is asking for an injunction that would require Google to obtain consent from publishers to use their website data to train its
general artificial intelligence products including Google’s own and those of rivals. It also asks Google to allow publishers who opt out of SGE to still show up
in Google search results, among other things.

The U.S. lawsuit follows an agreement Google reached last month with the Canadian government which would see the search giant paying Canadian media for use of
their content. Under the terms of the deal, Google will provide $73.5 million (100 million Canadian dollars) every year to news organizations in the country,
with funds distributed based on the news outlets’ headcount. Negotiations with Meta are still unresolved, though Meta began blocking news in Canada in August,
in light of the pressure to pay for the content under the new Canadian law.

The case also arrives alongside the filing of the U.S.
</item 18>

<item 19>
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
</item 19>

<item 20>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
The RSAs also motivate other companies to make or sell more Android devices, he said.

Expedia complains of too many ads on search, expensive ad payments

On November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on
organic listings.

“I must say I’m on the edge of revolt now that Google’s actions are so punitive, not just for Expedia but also for IAC and all the players that depend upon
something of a level playing field,” wrote Diller in a letter to Google back in 2019, according to Bloomberg.

Google CEO Sundar Pichai fired back that Google’s travel listings was one of the most popular experiences the company built.

Expedia execs also testified about the cost of ads and how increases had no impact on search results.
</item 20>

<item 21>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
Some critics, including Megan Grey, contend that Google uses semantic keyword matching to make searches less precise on purpose, which widens the pool of
content served and increases auctions for its ads. The quality tests, then, would ensure that the systems were still returning results that were justifiably
related to the input.
</item 21>

<item 22>
News publisher files class action antitrust suit against Google, citing AI’s harms to their bottom line (TechCrunch, 2023-12-15T17:56:02+00:00)
When online magazine The Atlantic modeled what would happen if Google integrated AI into search, it found that 75% of the time the AI would answer the user’s
query without requiring a click-through to its website, losing it traffic. This could have a major impact on publishers’ traffic going forward, as Google today
drives nearly 40% of their traffic, according to data from Similarweb.

Some publishers are now trying to get ahead of the problem. For example, Axel Springer just this week inked a deal with OpenAI to license its news for AI model
training. But overall, publishers believe they’ll lose somewhere between 20-40% of their website traffic when Google’s AI products fully roll out, The WSJ’s
report noted.

The lawsuit reiterates this concern, claiming that Google’s recent advances in AI-based search were implemented with “the goal of discouraging end-users from
visiting the websites of Class members who are part of the digital news and publishing line of commerce.”
</item 22>

<item 23>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
Bellshaw pulled up a message exchange in 2021 where Pichai asked for history to be turned off in a group chat. Pichai responded that he wanted to discuss a
personnel matter and the subject had nothing to do with the litigation hold, which he said he takes great care to comply with.

A moment of nostalgia for Internet Explorer

Cast your memories back to 2005, when Microsoft’s Internet Explorer became the default search engine. Back then, Google’s legal chief David Drummond sent
Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.

After establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The
letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”
</item 23>

<item 24>
5 things we learned so far about the Google antitrust case (TechCrunch, 2023-10-31T02:30:32+00:00)
He hoped to make Google and Apple’s services so connected as to be inseparable.

The way Pichai pitched it would have been a win-win for both companies. Google gets more people searching on its platform — not to mention all that juicy data —
and Apple would get more revenue, as a result of the lucrative search agreement the two signed.

For whatever reason, Cook didn’t take the bait. Apple doesn’t preload third-party software onto its devices, and it wasn’t going to make an exception for
Google.

Google’s deleted chat logs

During Pichai’s testimony, the DOJ touched on Google’s policy of deleting internal chat messages, despite being subject to a litigation hold. In February, the
DOJ accused Google of systematically destroying chats through its history-off option, which deletes messages every 24 hours unless a user manually changed the
setting.

Pichai said that he took action against the history-off default for chat in February to comply with the DOJ’s litigation hold, according to CNBC.
</item 24>

<item 25>
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
</item 25>

<item 26>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
In response, Google enacted a plan, which execs dubbed “Go Big in Europe,” to enhance search results in France and Germany in 2019 and 2020 with more local
content: news, post-game soccer video highlights, information on local television options for streaming, and pronunciation practice for different language. The
aim was to incentivize users to click on Google’s home screen, rather than the competition’s.

That revelation effectively supports the Justice Department’s argument that Google, without the push of competition, has little incentive to improve its
products, a classic consequence in a monopoly.

Mozilla digs in on Google’s supremacy in search

Interestingly, on November 1 Mozilla CEO Mitchell Baker provided a defense of Google’s quality as a search engine, even in “competitive” environments.
Specifically, Baker recounted how, Mozilla “failed” when it switched Firefox’s default search engine from Google to Yahoo.
</item 26>

<item 27>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair
strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its
devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell
Instagram and WhatsApp.

This isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group.
On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its
Android app store, Google Play, and its commission structure.

Now, onto the roundup!
</item 27>

<item 28>
More from the US v Google trial: vertical search, pre-installs and the case of Firefox/Yahoo (TechCrunch, 2023-11-13T14:37:44+00:00)
Before we go into some of these tidbits…

Why the Google vs. U.S. antitrust case matters

The government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering
competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its
behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer
manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can
improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that
antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law.
</item 28>

<item 29>
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
</item 29>

<item 30>
Is Google Search better than the rest? And is that fair? (The Age, 2023-10-22T18:00:00+00:00)
and it replied with more options that were all over $300. Another observation from this test is that I tend to use the Google Search bar as much for returning
to familiar sites as I do for searching for information, which means switching to another provider could be temporarily annoying simply because it wouldn’t
remember all the places I’ve been. Is Google as the default unfair?

In a response to the trial on Google’s blog, president of global affairs Kent Walker said the company’s search product was the favourite of users, browsers and
device makers simply because it was the best in the business. And being default was not determinative, he said; changing from Google Search to any other search
provider was easy, and the company competes with many others on phones and PCs. “People don’t use Google because they have to, they use it because they want
to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.”
</item 30>
=== END ITEM id=1b42ae5c4ed9bede ===
