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

=== ITEM id=ef6eca4eaa4a3f00 ===
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
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
AI was once laughed off. Now the richest men in Silicon Valley were shelling out billions to keep from being left behind. The lost ethics board

When Musk invested in DeepMind, he broke his own informal rule – that he would not invest in any company he didn’t run himself. The downsides of his decision
were already apparent when, only a month or so after his birthday spat with Page, he again found himself face to face with his former friend and fellow
billionaire. Larry Page, co-founder of Google. Credit: Bloomberg The occasion was the first meeting of DeepMind’s ethics board, on August 14, 2015. The board
had been set up at the insistence of the startup’s founders to ensure that their technology did no harm after the sale. The members convened in a conference
room just outside Musk’s office at SpaceX. But that’s where Musk’s control ended. When Google bought DeepMind, it bought the whole thing. Musk was out.
</item 1>

<item 2>
Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society (TechCrunch, 2023-10-31T17:21:47+00:00)
It won’t always be right, though. But it becomes the epicenter of where the conversation is. And then I’ll think what percentage of our population really wants
to earnestly take the breath and say, let me stop doomscrolling, and let me just wait until people can do the work, synthesize it and then help me react?” he
continued.

“We are also so plugged into all these storylines, whether it’s from traditional media, whether it’s from user-generated — someone on their phone, you know,
posting to a Telegram group. We are overwhelmed and overloaded, and it’s unfortunately appealing to the worst parts of a lot of our instincts,” Ohanian added.

In part, some of the issues around how social media platforms developed had to do with how the companies were originally developed. Ohanian, for example, shared
that Reddit grew out of his running of a forum site in college, when he believed that online community felt as real as offline.
</item 2>

<item 3>
Founders, are events useful? (TechCrunch, 2023-10-13T19:05:37+00:00)
A social social network network: Lauren reports that a former TikTok employee is building a social app for content creators to network and “spill the tea,” so
creators can help each other out making better, more engaging content.

X may go ad-free?: It appears that X (formerly Twitter) is testing three tiers of its Premium service, its CEO says. Under the hood, code shows one tier may be
ad-free. If it’s also troll free, please take my money right now.

Top reads on TechCrunch this week

A lot of amazing news on the site this week, but if we go by the raw numbers, here are the most popular stories — the ones that I didn’t already cover above,
that is.

Passwords? We don’t need no steenkin’ passwords: Passkeys are a phishing-resistant alternative to passwords that allow users to sign in to accounts using the
same biometrics or PINs they use to unlock their devices or with a physical security key. Google is now making it the default sign-in method for all users.
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
How the OpenAI fiasco could bolster Meta and the ‘open AI’ movement (TechCrunch, 2023-11-21T15:50:50+00:00)
On the surface, Microsoft is currently looking like the biggest winner amidst the OpenAI turmoil, as it was already apparently looking to reduce its reliance on
OpenAI even though it remains once of its major shareholders. But Facebook’s parent Meta could also stand to benefit, as businesses pursue multi-modal
strategies or models with a more “open” ethos embedded.

“Open source today offers a wide variety of models for companies to essentially diversify,” Ceze added. “By doing so, these startups can quickly pivot and
minimize risk. There is also a major upside — many of these models already outperform the likes of OpenAI’s in terms [of] price-performance and speed.”

A leaked internal memo from Google earlier this year seemed to express fears that despite the huge advances made by proprietary LLM models from the likes of
OpenAI, open source AI would ultimately trump them all. “We have no moat, and neither does OpenAI,” the document noted.
</item 5>

<item 6>
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
</item 6>

<item 7>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter. Hassabis and his co-founders
insisted on two conditions: No DeepMind technology could be used for military purposes, and its AGI technology must be overseen by an independent board of
technologists and ethicists.

Google offered $US650 million. Zuckerberg of Facebook offered a bigger payout to DeepMind’s founders but would not agree to the conditions. DeepMind sold to
Google. Zuckerberg was determined to build an AI lab of his own. He hired Yann LeCun, a French computer scientist who had also done pioneering AI research, to
run it. A year after Hinton’s auction, Zuckerberg and LeCun flew to Lake Tahoe for the same AI conference. While padding around a suite at the Harrah’s casino
in his socks, Zuckerberg personally interviewed top researchers, who were soon offered millions of dollars in salary and stock. AI was once laughed off.
</item 7>

<item 8>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
When Google bought DeepMind, it bought the whole thing. Musk was out. Financially, he had come out ahead, but he was unhappy. Three Google executives now firmly
in control of DeepMind were there: Page; Sergey Brin, a Google co-founder and Tesla investor; and Eric Schmidt, Google’s chair. Among the other attendees were
Reid Hoffman, another PayPal founder; and Toby Ord, an Australian philosopher studying “existential risk”.

The DeepMind founders reported that they were pushing ahead with their work but that they were aware the technology carried serious risks. Suleyman, the
DeepMind co-founder, gave a presentation called “The Pitchforkers Are Coming.” AI could lead to an explosion in disinformation, he told the board. He fretted
that as the technology replaced countless jobs in the coming years, the public would accuse Google of stealing their livelihoods. Google would need to share its
wealth with the millions who could no longer find work and provide a “universal basic income,” he argued. Musk agreed.
</item 8>

<item 9>
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
</item 9>

<item 10>
With Amo, the founder of Zenly wants to make social apps social again (TechCrunch, 2023-11-19T12:07:20+00:00)
With Amo, the founder of Zenly wants to make social apps social again The startup raised $20 million and is launching ID today

In 2018, I wrote a TechCrunch article stating that 2018 was “the year social networks were no longer social.” Reflecting on that article, I’m not sure that 2018
was the turning point. But the premise of the article still holds up well.

At some point, social networks were no longer about connecting with your closest friends, keeping up with long-distance family members and feeling a special
connection with people you love.

TikTok, Instagram, Snapchat, Facebook, YouTube and X (formerly Twitter) are all slowly evolving to become the same infinite scrollable feed of
algorithm-optimized short videos from top performers.

And it turns out that I’m not the only one who has noticed that social networks have been slowly drifting away from their original purpose.
</item 10>
=== END ITEM id=ef6eca4eaa4a3f00 ===

=== ITEM id=51739a90f4c4a2a5 ===
Gold sentences and facts:
Sentence 0: 10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late.
  Fact 0.0: At 10:59 p.m., the Cowboys are moving the ball.
  Fact 0.1: The Cowboys are moving the ball when it is a bit too late.
Sentence 1: Mixon has been forced to step up without Burrow under center, doubling as a runner and receiver for Cincinnati.
  Fact 1.0: Mixon has been forced to step up.
  Fact 1.1: Burrow is not under center.
  Fact 1.2: Mixon is forced to step up without Burrow under center.
  Fact 1.3: Mixon doubles as a runner for Cincinnati.
  Fact 1.4: Mixon doubles as a receiver for Cincinnati.

Retrieved items:
<item 1>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
They'll start the second half with the ball.

9:56 p.m. — 49ers edge rusher Drake Jackson is walking off after requiring some medical attention on the field.

9:52 p.m. — The 49ers tried to put something together, but a third down pass to Aiyuk was too low. Dallas has a minute to try and get something before halftime.

9:45 p.m. — The Cowboys looked better on third down on their last drive, but Prescott's third down pass to Ferguson is short of the line to gain and ends this
drive quickly. The 49ers will have a couple minutes to blow this game open before halftime.

49ers 21, Cowboys 7

9:38 p.m. TOUCHDOWN — McCaffrey pushes through for the touchdown. That's his 14th consecutive game finding the end zone, and it puts the 'Niners back up two
scores.

9:37 p.m.
</item 1>

<item 2>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
This 49ers defense has picked up right where it left off against the Cowboys in January.

8:44 p.m. — Dallas' second drive starts out with Tony Pollard getting dropped for a loss.

8:40 p.m. — Big-time stop by the Cowboys' defense as Johnathan Hankins deflects Purdy's third down pass with his helmet and forces a punt.

8:36 p.m. — The 49ers' defense was all over Dallas on that drive. Three plays, two incompletions, and now the Cowboys will punt it right back.

49ers 7, Cowboys 0

8:30 p.m. TOUCHDOWN — Purdy finds a wide open Kittle in the end zone for the score! That was a well-scripted opening drive by Kyle Shanahan, and San Francisco
has an early 7-0 lead.

8:27 p.m.
</item 2>

<item 3>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Not allowing even a field goal is a huge win for Dallas.

8:56 p.m. — The 49ers are excelling on third down early in this game, and Purdy flips it to Kyle Juszczyk to keep the Niners' drive alive on 3rd & 1.

8:54 p.m. — Now Kearse lines up offsides... That extends the drive for the 49ers, who aren't moving as freely as they were on that opening drive.

8:50 p.m. — Jayron Kearse comes through with the Cowboys' first sack of the night, but Deebo Samuel makes up 11 of the 12 yards San Francisco needs on the next
play.

8:46 p.m. — There's the first catch of the night for Lamb, but he doesn't get the first down and Dak Prescott is sacked by Nick Bosa to quickly end the drive.
This 49ers defense has picked up right where it left off against the Cowboys in January.
</item 3>

<item 4>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
End of first quarter: 49ers 7, Cowboys 0

9:04 p.m. — Confirmed: 49ers ball. San Francisco has an instant opportunity here.

9:01 p.m. FUMBLE — Oh no! Pollard gets the ball knocked out deep in the Cowboys' own territory, and it looks like it was recovered by the 49ers. Officials are
checking to make sure nobody was out of bounds, but that ball was very much out of Pollard's control. The 49ers could be set up in the red zone.

8:59 p.m. — The drive completely falls apart for the 49ers. A delay of game penalty sets San Francisco back, and Christian McCaffrey is dropped for a five-yard
loss on 3rd & 15. Not allowing even a field goal is a huge win for Dallas.

8:56 p.m.
</item 4>

<item 5>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Cowboys have a first down! And it only took 17+ minutes. Dallas is still operating in its own territory, but it feels like points are needed on this drive
to keep the game within reach.

49ers 14, Cowboys 0

9:14 p.m. TOUCHDOWN — That was quick! Purdy hits Kittle after some trickery, and it's a two-score night for the tight end. 49ers lead, 14-0.

9:13 p.m. — The Cowboys are forced to punt out of their own end zone after Fred Warner sacks Prescott. Warner now has a sack and a fumble. The 49ers will have
strong field position.

9:10 p.m. FUMBLE — Oh no again! They're trading fumbles! Jourdan Lewis takes the ball right out of McCaffrey's arms and spoils the 49ers' hopes of making this a
14-0 game.
</item 5>

<item 6>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
— The Purdy-to-Aiyuk connection continues to get stronger. Aiyuk goes for 23 yards to give the 49ers some breathing room after opening their drive with a false
start.

49ers 21, Cowboys 10

10:17 p.m. FIELD GOAL — The 49ers' defense tightens the screws and limits Dallas to a field goal. Brandon Aubrey drills it from 50 yards out, and it's down to a
21-10 lead for San Francisco.

10:13 p.m. — Prescott hits CeeDee Lamb for a big 29-yard gain. That quickly takes the Cowboys into 49ers territory to start the second half.

End of first half: 49ers 21, Cowboys 7

9:58 p.m. — San Francisco's defense is all over the Cowboys yet again. It's Dre Greenlaw's turn to sack Prescott, and Dallas is waving the white flag on the
half.
</item 6>

<item 7>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
11:00 p.m. INTERCEPTION — So much for that. Oren Burks gets the 49ers' third interception of the night as Prescott continues to struggle against this San
Francisco defense.

10:59 p.m. — The Cowboys are moving the ball now, when it's a bit too late. Tony Pollard rips off a 26-yard run to finally break through.

49ers 42, Cowboys 10

10:53 p.m. TOUCHDOWN — And just like that, Jordan Mason runs in for another 49ers touchdown! This is ugly on all levels for the Cowboys, and Leighton Vander
Esch is now down as well. 42-10, 49ers.

10:52 p.m. INTERCEPTION — Fred Warner now has a sack, forced fumble and an interception tonight. A pass from Prescott is deflected into his hands, and the 49ers
have another instant scoring opportunity.
</item 7>

<item 8>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
Jerry Jones said earlier in the week that Sunday night's game would show him how the Cowboys stack up against the NFL's best.

The answer? Not very well.

Everything that could go wrong for Dallas on Sunday did go wrong, in a game that confirmed a 49ers team that ended the Cowboys' season in each of the last two
years remains a level above its NFC counterpart.

The 49ers carved up Dallas' defense on the opening drive, and that was a sign of what was to come. Brock Purdy threw four touchdowns in the 42-10 rout,
including three to George Kittle. Christian McCaffrey scored a touchdown for a 14th consecutive game as all of San Francisco's offensive weapons got involved.

MORE: Comparing stats for NFC stars Nick Bosa, Micah Parsons

Defensively, the 49ers were relentless. Nobody had a larger impact than Fred Warner, who finished the night with an interception, sack and forced fumble in a
swarming performance.
</item 8>

<item 9>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
The Cowboys' offense looked very much like it did in last season's playoff exit -- uncomfortable and out of sorts. Dallas was sluggish through the first half of
the game, and the night unraveled quickly when Dak Prescott threw three interceptions in the second half. It got out of hand so quickly that Sam Darnold entered
the game for the 49ers with more than 12 minutes remaining.

For Prescott and the Cowboys, questions about whether they can get over the hump against the class of the NFC will persist. While their wins have been dominant,
Dallas already sits two games behind the Eagles in the NFC East.

The 49ers are rolling right along, with all of their major offensive weapons now healthy and the defense living up to its billing as well. San Francisco, now
5-0, doesn't face a team with a winning record until Week 10, though the Browns, Vikings and Bengals are far from automatic wins for Kyle Shanahan's team.
</item 9>

<item 10>
Cowboys vs. 49ers live score, updates, highlights from NFL 'Sunday Night Football' game (Sporting News, 2023-10-08T22:47:03+00:00)
For a Cowboys team that has embarrassed a few opponents this year, they might be on the bad end of an ugly game here.

49ers 28, Cowboys 10

10:27 p.m. TOUCHDOWN — Not one, not two, THREE touchdowns for George Kittle. The 49ers have everything working offensively tonight, and any Kittle fantasy
owners are thrilled.

10:25 p.m. — Aiyuk's loss is Deebo Samuel's gain. Samuel goes 43 yards on the very next play, which tells you how automatic the 49ers' offense is right now. San
Francisco is in the red zone looking to make this a three-score game.

10:23 p.m. — An offensive holding call wipes out a 40-yard gain from Aiyuk. Wow. That's a momentum swing. Spencer Burford is the culprit.

10:21 p.m. — The Purdy-to-Aiyuk connection continues to get stronger.
</item 10>
=== END ITEM id=51739a90f4c4a2a5 ===

=== ITEM id=66ff75ca17c08858 ===
Gold sentences and facts:
Sentence 0: On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential ground invasion.
  Fact 0.0: On Oct. 12, 2023, Israel warned Gaza residents.
  Fact 0.1: The warning was given to 1.1 million Gaza residents.
  Fact 0.2: The warned residents were in the northern section of the enclave.
  Fact 0.3: The warning told the residents to leave for the southern region.
  Fact 0.4: The warning was given in advance of a potential ground invasion.
Sentence 1: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 1.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.

Retrieved items:
<item 1>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
In an article five years ago, Ortal, currently in active military service an unavailable for interview, wrote, "Israeli restraint as regards proactive thwarting
of offensive capabilities in the Gaza Strip stems from a clear Israeli strategy that strives to reduce the influence of the enemy on the current routine of life
in Israel, even at the price of worsening of the threat in the future." Ortal pointed out that this policy was natural and that "a Western nation cannot lead
its life in an endless state of emergency. But alongside the understanding of the need for a containment policy, it must be recognized that the risks are high,
as evidenced by the steeply worsening intensity of conflicts between Israel and the terrorist organizations in Gaza."

According to Ortal, in recent decades, Hamas and Hezbollah developed ballistic missiles and terror tunnels in response to Israeli policy that sought minimal
friction with the enemy through separation and withdrawal from the security zone in southern Lebanon, and the Gaza Strip.
</item 1>

<item 2>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Truly disturbing acts of terror directed at civilians".

Some of the images coming out of Sderot over the last few hours are reaching degrees of horror that I dont really think I can use words to properly describe.
Truly disturbing acts of terror directed at civilians. — OSINTtechnical (@Osinttechnical) October 7, 2023

Commenting on the coordinated attack, Rania Khalek, a journalist, said that the strategic planning that must have gone into this against "one of the most
high-tech occupation and surveillance systems in the world" will be studied by many. "The Israelis must be totally losing it right now."
</item 2>

<item 3>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas is an excuse, new lands are wonderful



I definitely support Israel pic.twitter.com/AbLdeazQjR October 7, 2023

Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that
infiltrated its territory. "The second objective, at the same time, is to exact an immense price from the enemy, within the Gaza Strip as well," he said, adding
that the third objective was to reinforce other fronts so that nobody should mistakenly join this war. "We are at war. In war, one needs to be level-headed. I
call on all citizens of Israel to unite in order to achieve our highest goal – victory in the war," Netanyahu said.

Since this morning, the State of Israel has been at war. Our first objective is to clear out the hostile forces that infiltrated our territory and restore the
security and quiet to the communities that have been attacked.
</item 3>

<item 4>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas' surprise attack on Israel has shocked many who are now questioning the country's famed intelligence agency Mossad, which is known for its
intelligence-gathering capability. For some, the belief that Mossad was among the best in the spying world was shattered today when Hamas carried out the
biggest coordinated attack in Israel and captured some territory in southern Israel. It is not the attack that has surprised people but the scale at which Hamas
operated and got control of some military bases and soldiers.

Also Read: Israel at war: Hamas cells, military compounds destroyed, US backs Tel Aviv; all you need to know

Younis Tirawi, who reports on Palestinian Affairs, said that Hamas militants had taken over Sderot police HQ and armored vehicles in the town. An analyst wrote
that some of the images coming out of Sderot over the last few hours were reaching degrees of horror "that I don't really think I can use words to properly
describe.
</item 4>

<item 5>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
The second objective, at the same time, is to exact an… pic.twitter.com/MzKs7tfv4M — Benjamin Netanyahu - בנימין נתניהו (@netanyahu) October 7, 2023

Reacting to this, however, a user said that knowing the proficiency of the Mossad he found the idea of a "surprise attack" to be a willing lie.

Well, you got the war you wanted Beni...at what cost of Israeli lives you don't care. Knowing the proficiency of the Mossad I find the idea of a "surprise
attack"to be a willing lie on your side of this. https://t.co/35qRvgBJt2 — timm (@timmainsson) October 7, 2023

Ben Muasya, a journalist, said that Mossad is overrated.
</item 5>

<item 6>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"The Israelis must be totally losing it right now."

the strategic planning that must have gone into this against one of the most high tech occupation and surveillance systems in the world will be studied by many,
the Israelis must be totally losing it right now https://t.co/oUX2rokEjh — Rania Khalek (@RaniaKhalek) October 7, 2023

Tarik Cyril Amar said: "And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed."

And the intel, and the special forces training... both areas where Israel mythology will be very upset indeed. — Tarik Cyril Amar (@TarikCyrilAmar) October 7,
2023

Mjoachim, another user, said that Israel's defence forces have lost their fame. He said the IDF lost an army base, a general was captured right out of his
house.
</item 6>

<item 7>
Israel's defense undone by reliance on technology (Globes English | Israel Business Arena, 2023-11-05T13:35:33+00:00)
On Saturday, October 7, for example, suspicious movement in Gaza convinced the Shin Bet head to send a special team to the southern border, but he was unable to
convince the top IDF echelons to prepare accordingly. To handle those weak signals, Eiland claims, one must use "scenario theory", meaning, taking into account
scenarios that have a low probability of occurring, but with devastating results should they occur, and the simple way to prepare for them.

Eiland explains: "You detect certain noises on Friday night, but conclude that the probability of an attack is low, let's say only 10%. Those who act only on
the probability do not consider the question: if this attack happens - will you be able to live with yourself for eliminating this threat? There’s also a third
consideration, which is ease of effort: what is the effort required of me to prepare for this unlikely but dangerous scenario?
</item 7>

<item 8>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
"How come they never knew Hamas militants planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate heavily? We wait to see how it goes
but it will be messy."

Mossad is overated how comes they never knew this Hamas militants and had planned a sudden attack on Israel? Or was it a trap to get a reason to retaliate
heavily?We wait to see how it goes but it will be messy. pic.twitter.com/W5vpoylus2 — Ben Muasya. (@muasya_ben) October 7, 2023

Another user said that no one can carry out such a major attack against Israel without intelligence support. "Who gave Hamas Intelligence? It was as if the
entire Israeli army and the Mossad were asleep."
</item 8>

<item 9>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
BHARAT is not immune to #terrorism and internal enemies @HMOIndia @PMOIndia @AmitShah — Dr. Jaswant Gandhi 🇮🇳 (@JaswantDr) October 7, 2023

However, some speculated that Mossad knew what was going to happen but let Hamas make a mistake so that Israel could launch a ferocious attack on it. "Mossad
was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, but they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful," a user said.

#Mossad was not so ignorant that it did not know about this #attack. Israel wanted to invade Palestine, they had no excuse, stupid #Hamas gave them the reason.
Hamas is an excuse, new lands are wonderful
</item 9>

<item 10>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
He said the IDF lost an army base, a general was captured right out of his house. "Invincible "Merkava MK4 MBT's captured, and destroyed...what is wrong with
the Israeli security and defence forces? And (what) MOSSAD did?"

Dr Jaswant Gandhi, a professor of management, said that attack on Israel was a "great Mossad failure". He said India had to learn a lot and keep all
intelligence agencies duly greased and internecine conflict-proof. "Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence
Agencies duly greased and internecine conflict-proof. BHARAT is not immune to #terrorism and internal enemies," he wrote on X.

#Israel_under_attack Great #Mossad failure. GoI has to learn a lot and keep all Intelligence Agencies duly greased and internecine conflict proof.
</item 10>
=== END ITEM id=66ff75ca17c08858 ===

=== ITEM id=b7170e11c3f3c3d1 ===
Gold sentences and facts:
Sentence 0: Another user said that no one can carry out such a major attack against Israel without intelligence support.
  Fact 0.0: Another user said that no one can carry out such a major attack against Israel without intelligence support.
Sentence 1: Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground invasion.
  Fact 1.0: Israel stopped allowing deliveries of food to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 1.1: Israel stopped allowing deliveries of fuel to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 1.2: Israel stopped allowing deliveries of other supplies to Gaza's 2.3 million residents on Oct. 10, 2023.
  Fact 1.3: Israel is reportedly preparing for a ground invasion.

Retrieved items:
<item 1>
Israel thwarts Hamas on the financial battlefield (Globes English | Israel Business Arena, 2023-11-19T19:05:30+00:00)
RELATED ARTICLES Tech experts battle to shut down Gaza financial pipeline

"Overnight, we started receiving hundreds of reports about unusual activity by customers of financial institutions in the State of Israel," she says. Before
October 7, IMPA received about 100 reports a day dealing with terror financing. That number jumped to about 1,000. "There is a 900% increase in reports on
terror financing, and we don't waste time, everything happens very, very quickly. As we receive the information, we simultaneously pass it on to the operational
professionals who locate the money pipelines, and block them. Some information will not necessarily lead to terror financing, but we don’t have time to waste.
Bank CEOs call and draw our attention to the information they convey to us that warrants special attention."

Has information from banks and financial institutions led to blocking the money pipeline for terrorism?

"There was very high-quality intelligence that we passed on to the security agencies. We don't always know the results."
</item 1>

<item 2>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
may again get impacted on account of higher input costs and higher freight and fuel costs.

Rural India is a key engine that is not firing at the moment. Valuations also are rich and do not offer margin of safety in the near to medium term. Near-term
attractiveness of equities will reduce, as investors will demand a higher risk premium. The Indian stock market will likely correct materially, as FIIs look to
withdraw funds from risk assets and seek safe havens in US Dollar, US treasuries offering high yields and Gold.

Sectors which will be negatively impacted include Petrochemicals, Oil Marketing, Paints, Packaging, Textiles and Chemicals.

The writer is Group President & Head - Institutional Equities, YES Securities India

SHARE Copy link

Email

Facebook

Twitter

Telegram

LinkedIn

WhatsApp

Reddit
</item 2>

<item 3>
Israel’s startup ecosystem: down but not out (TechCrunch, 2023-10-13T19:10:07+00:00)
The world is still coming to terms with Hamas’ deadly attacks on Israelis last weekend — and everything else that has unfolded so far in the aftermath,
including the barrage of retaliation strikes on Gaza.

Undoubtedly, the ramifications for the technology ecosystem pale in comparison to the human toll that the fighting is taking, on both sides. But when
considering how the wider region will weather the destruction, and one day recover, it’s an important industry to consider. Please also read here about how this
is playing out for Palestinians in Gaza and beyond. And a below, we show how a look at how the situation is playing out in the startup ecosystem in Israel sheds
light on its impact to the country as a whole.

In such a small country, just about everyone knows someone who was directly affected by the attacks, or someone involved in the subsequent defense and
retaliation — and often all three.

Technology is, without question, a giant part of Israel’s economy.
</item 3>

<item 4>
ASX set to rise ahead of inflation report after big tech lifts Wall Street (The Sydney Morning Herald, 2023-10-24T20:05:06+00:00)
Concerns about the conflict spreading more broadly have eased amid growing calls within Israel to rethink a ground invasion of Gaza because of fears about the
fate of some 200 hostages held there, the danger of retaliation by Hezbollah and the risk of Israeli military casualties. Tweet of the day Quote of the day “I
think when you get to those decisions, you can then sit on them for a while… or you can say let’s just deal with it [and] that’s what we have done today,” said
Magellan chairman Andrew Formica as the group’s chief executive stepped down abruptly amid the company’s efforts to turn around its performance.
</item 4>

<item 5>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
India shipped goods around $8billion to Israel. Petroleum products account for 65 per cent of the exports, followed by gems and jewellery (15 per cent) and
electrical machinery (3 per cent). Trade can be adversely impacted if operations at the Israeli ports are disrupted. Fertiliser movement from Israel’s Port of
Ashdod accounts for 3 per cent of global potash supply. Any disruption on this front will also be a dampener.

But, what if the current hostilities turn into a full-scale conflict?

What supports this proposition is the fact that while there have been many conflicts between Israel and Hamas in the past decades, there have been none of this
intensity. With at least 1,200 Israelis killed, this has been the biggest single-day killing of Jews since Holocaust, and Israel opposition is also united in
retaliating. Israel has termed it a ‘War’, formed a war management cabinet with a key opposition member, and vowed to change the look of the ‘Middle East’.
</item 5>

<item 6>
Dissecting impact of Israel-Hamas conflict on markets  (Business Line, 2023-10-14T15:20:36+00:00)
The world can ill-afford another full-fledged war, when economies are already battling high inflation and interest rates, an ongoing Russia-Ukraine war and now,
a slowing China. This is precisely why many countries are trying to restrict any escalation in the ongoing Israel-Hamas conflict. The non-escaltion of the
current hostilities seems to be the base case presumed by the Indian stock market presently, going by the market movement in the last week.

The US treasury secretary also stated that the current conflict is not likely to significantly impact global economic outlook. History too shows that the these
conflicts tend to have limited impact on markets. We studied the history of conflicts between Israel and Hamas, and found that these tend to have limited impact
on global financial markets, across crude oil, US 10-year yields, US equities and market volatility (see charts).

However, as far as trade goes, there may be some impact. In 2022-23, Israel’s share in exports from India stood at 1.87 per cent.
</item 6>

<item 7>
"There are no sacred numbers" (Globes English | Israel Business Arena, 2023-10-12T18:50:29+00:00)
The first week of fighting in the bitter war in Israel’s south is almost over, with the financial markets trying to gauge the damage that the Israeli economy
will sustain. Clearly, we are talking about a very significant event, but the fog of war makes it hard to estimate its full consequences. The main index on the
Tel Aviv Stock Exchange, the Tel Aviv 35, fell 6.4% over the week.

Prof. Leo Leiderman, chief economic adviser at Bank Hapoalim and a professor of economics at Tel Aviv University, sets out for "Globes" his proposals for
rehabilitating the economy and repairing the damage that the war will cause.

What has been going through your mind in these first days of the war?

"What most preoccupies me is not the economy but concern for the families of the soldiers, the wounded, and the captives, and solidarity with them. I have four
relatives from Kibbutz Nir Oz who are still missing, and we have no information about them.
</item 7>

<item 8>
&#039;Massive intel failure by Mossad&#039;: Hamas&#039; surprise attack on Israel shocks internet (Business Today | Latest Stock Market And Economy News India,
2023-10-07T15:37:43+00:00)
Hamas is an excuse, new lands are wonderful



I definitely support Israel pic.twitter.com/AbLdeazQjR October 7, 2023

Israeli Prime Minister Benjamin Netanyahu said that since this morning, Israel had been at war and its first objective was to clear out the hostile forces that
infiltrated its territory. "The second objective, at the same time, is to exact an immense price from the enemy, within the Gaza Strip as well," he said, adding
that the third objective was to reinforce other fronts so that nobody should mistakenly join this war. "We are at war. In war, one needs to be level-headed. I
call on all citizens of Israel to unite in order to achieve our highest goal – victory in the war," Netanyahu said.

Since this morning, the State of Israel has been at war. Our first objective is to clear out the hostile forces that infiltrated our territory and restore the
security and quiet to the communities that have been attacked.
</item 8>

<item 9>
Israel's blockade of Gaza means that a region the size of Philadelphia faces a real risk of starvation, says Peace Studies scholar (Fortune,
2023-10-13T19:50:48+00:00)
International aid groups are warning that they cannot deliver food and other basic services to people in the Gaza Strip and that a “dire” humanitarian crisis is
set to worsen.

International aid groups provide food and other means of support to about 63% of people in Gaza.

Israel stopped allowing deliveries of food, fuel and other supplies to Gaza’s 2.3 million residents on Oct. 10, 2023, and is reportedly preparing for a ground
invasion.

On Oct. 12, 2023, Israel warned 1.1 million Gaza residents in the northern section of the enclave to leave for the southern region, in advance of a potential
ground invasion.

I am a scholar of peace and conflict economics and a former World Bank consultant, including during the 2014 war between Hamas and Israel.
</item 9>

<item 10>
Palestine’s growing tech industry has been literally blown apart by Israel’s war on Hamas (TechCrunch, 2023-10-14T07:46:27+00:00)
Sturgill said that as well as the intensely difficult conditions in Gaza, which is being hit by Israeli missiles, the situation in Ramallah is “super tense. I
feel like the situation is going to get significantly worse there over the coming weeks.”

Iliana Montauk, co-founder and CEO of Manara — a social impact startup funded by Y Combinator, Seedcamp, Reid Hoffman, Eric Ries, Marc Benioff, Paul Graham, and
Jessica Livingston, among others — told TechCrunch via email that connectivity has decreased significantly in the past 24 hours.

“Though Gaza has been bombed many times before, this time is completely different for the tech sector for several reasons. Electricity was cut off to the entire
[Gaza] strip. A significant amount of infrastructure has been bombed (including both ISPs and many tall apartment buildings that hold cell phone towers). Entire
middle-class neighborhoods are being destroyed.”
</item 10>
=== END ITEM id=b7170e11c3f3c3d1 ===

=== ITEM id=0027d7438a724c0e ===
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
And when we tried to bundle Fortnite with other smartphone manufacturers like OnePlus and carriers of all sorts, they told us they couldn’t do a deal because
Google had done a secret deal with them.

It was really disconcerting to see the extent of bad faith efforts that were going on in a company of Google’s size. You’d think a trillion-dollar company would
develop to the point where they have pretty respectable processes and leadership structures that provide a check and balance against wrongdoing, but they were
rampantly destroying all their chats on these topics.

You’d see long conversation threads would start to get into a spicy antitrust issue, and suddenly somebody points out the history’s on and the chat goes silent.
They just turned it off to have the documents destroyed. It was great to see that all called out in detail.

Any particular evidence where you’re like, “Oh wow, I never saw that, and now that specific thing came out for the world to see”?

As an employee of Epic, I’ve not been able to see Google’s internal documents until the trial started.
</item 1>

<item 2>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
(Google told The Verge over email that this refers to how many users have enabled the Android sideloading flow, not necessarily followed through with an
install.) “A billion people have done it after getting notified of the potential risks,” Pomerantz said. “That’s because Android users have a real choice.”

Google also took its own turn trying to paint Epic as the bad guy. First, it pointed out how Epic hatched a secret plan called “Project Liberty” to quietly
update Fortnite with code to bypass app store fees, get its app kicked off Apple’s and Google’s app stores, and sue.

Then, it showed off a few out-of-context quotes from internal Epic communications — suggesting that phrases like “How do we not look like the bad guys?” and
“Just planting the nefarious seed now” and “I mean everything we’re attempting is technically a violation of Google’s policy, right?” showed that Epic knew it
was breaking bad at the time it did the deed.
</item 2>

<item 3>
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
</item 3>

<item 4>
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
</item 4>

<item 5>
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
</item 5>

<item 6>
Epic v. Google, explained (The Verge, 2023-11-05T11:00:00+00:00)
Google will say that every Samsung Galaxy phone has “a competing app store right there on the homescreen,” even though Google apparently tried to quash that
store, too, fearing it could cost the company billions.

We caught Google trying to play the China card in a pre-briefing with press, reminding us that Epic is 40 percent owned by Chinese company Tencent — even though
Google itself allegedly considered teaming up with Tencent to eliminate the Epic threat. While the parties can’t argue it in court, Google can attempt to subtly
play the China card: “Google may ask a witness once about Tencent’s domicile,” reads one of Judge Donato’s orders.

In the same press briefing, Google suggested to journalists that Epic should have joined the company’s “User Choice Billing” program if it wanted an alternative
payment system, instead of suing. But that program didn’t exist until 2022, long after the lawsuit, and it only knocks 4 percentage points off Google’s fee.

When should we expect a jury verdict?
</item 6>

<item 7>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
showed that Epic knew it was breaking bad at the time it did the deed.

But Epic mentioned Project Liberty in its own opening statement — so, by that point, it had already been an hour since it admitted it intentionally broke
Google’s rules. “Epic decided to stand up because that’s what you do to a bully,” Bornstein told the jury.

“All we know is whatever is in the destroyed chats, as bad as the documents are, is worse.”

And it’s possible no examination will be able to take the stink off one of Google’s ugliest moves: the one where Google employees up to and including CEO Sundar
Pichai were caught setting sensitive chats to auto-delete to keep them out of a court’s hands. The court has already decided Google should be sanctioned in some
way for making potential evidence disappear, and Bornstein used it to plant persistent seeds of doubt in the minds of jury members. “All we know is whatever is
in the destroyed chats, as bad as the documents are, is worse.
</item 7>

<item 8>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This wasn’t the only layoff round of the year. Unity lowered its headcount three times in 2023, affecting about 900 jobs. In its quarterly financial results in
November, Unity reported a yearly revenue increase of 69 percent and it told investors, “We continued to manage costs well.”

Sony cut 100 jobs at Bungie, a company it bought for $3.6 billion in 2022. According to developers that are still there, Sony executives are attempting to use
this upheaval to wrest more control of the studio from Bungie founders and leaders.

Epic Games fired roughly 830 people this year, or 16 percent of its staff. This included significant job cuts at Mediatonic, the studio behind Fall Guys that
Epic purchased in 2021.

Mediatonic

“For a while now, we've been spending way more money than we earn,” CEO Tim Sweeney wrote about the layoffs.
</item 8>

<item 9>
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
</item 9>

<item 10>
The Epic v. Google trial may come down to simple v. complicated (The Verge, 2023-11-07T15:00:40+00:00)
The upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and /
or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as
the bad guy on day one.

It’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision
from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision
would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever”
to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure
Google’s opening statement countered it.
</item 10>
=== END ITEM id=0027d7438a724c0e ===
