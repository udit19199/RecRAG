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

=== ITEM id=ec9c4db9d9d6dffe ===
Gold sentences and facts:
Sentence 0: Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9).
  Fact 0.0: Tyreek Hill had two touchdowns in this game.
  Fact 0.1: Tyreek Hill had 157 receiving yards in this game.
  Fact 0.2: Tyreek Hill helped the Miami Dolphins rout the Washington Commanders.
  Fact 0.3: The Washington Commanders were 4-9 at this point.
Sentence 1: • Six catches for 163 yards and a touchdown against Carolina left the Dolphins’ Tyreek Hill with 812 yards through the first six games.
  Fact 1.0: Tyreek Hill made six catches for 163 yards and a touchdown against Carolina.
  Fact 1.1: Tyreek Hill had 812 yards through the first six games.
Sentence 2: Of Miami's five remaining opponents, four have pass defenses ranked in the league's top eight.
  Fact 2.0: Miami has five remaining opponents.
  Fact 2.1: Four of Miami's five remaining opponents have pass defenses ranked in the league's top eight.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning in three limited practices this week, we'll trust that Hockenson is ready to go, especially since he continued to play through the ailment in Week
9 as a big part of the offense with Josh Dobbs under center.

As long as Hockenson is playing, he's not being benched in any fantasy format. The tight end projects as a top option at the position and remains the Vikings'
leader in targets with 80 on the season. Dobbs enjoyed targeting the tight ends in Arizona and continued that trend in his Minnesota debut. The floor for
Hockenson is equivalent to that of a rooftop bar.

WEEK 10 DFS: DK lineup | FD lineup | Best values | Best stacks

Is Curtis Samuel playing Week 10?

UPDATE: Samuel is officially ACTIVE.

Samuel (toe) is expected to play on Sunday when the Commanders take on the Seahawks at 4:25 p.m. ET, according to ESPN's Jeremy Fowler.
</item 1>

<item 2>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
After turning back the clocks last week, it now gets darker earlier for the foreseeable future. That darkness allows us to see the stars on a clear night, so
it's only natural to wonder if the pass-catching stars will shine bright in Week 10 of the NFL season. We already know Tee Higgins and Nico Collins are OUT,
while Justin Jefferson, Ja'Marr Chase, and T.J. Hockenson are among the other star pass-catchers beaten up heading into the week. Can we expect them on the
field when the action kicks off Sunday?

Below, we'll give you the latest news on these ailing pass-catchers and break down how their injuries could potentially affect the Week 10 fantasy landscape.

WEEK 10 INJURY UPDATES: Early WRs | RBs

Is Justin Jefferson playing Week 10?

UPDATE: The Vikings will not activate Jefferson before Sunday's game and have officially ruled him OUT, according to NFL Network's Tom Pelissero.
</item 2>

<item 3>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
It can be tough to trust Taysom Hill (vs. Panthers) in full-point PPR leagues, but a matchup against the league's second-worst run defense gives him a high
ceiling regardless of format. He's been more involved as a receiver this year, so he'll get a few catches to go along with the rushing stats, too. With Derek
Carr (concussion) banged up, Hill might throw more this week, too.

Gerald Everett (vs. Broncos) and Isaiah Likely (vs. Rams) have matchups against bottom-six TE defenses, and Cole Kmet (vs. Lions), Jake Ferguson (vs. Eagles),
and Kyle Pitts (vs. Buccaneers) also have above-average matchups that give them higher floors and higher ceilings than usual.

For those in deeper leagues, Tyler Conklin (vs.
</item 3>

<item 4>
DraftKings Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments (Sporting News, 2023-12-07T22:18:00+00:00)
If they're trailing, JSN should see action with a negative game script.

WEEK 14 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

TE Brevin Jordan, Texans @ Jets ($3,100). Jordan was a great streaming option and fill-in for the crowd that lost Dalton Schultz (hamstring) on Sunday, hauling
in three-of-four targets for 64 yards against the Broncos. If Schultz is out again, Jordan is a serviceable option that has touchdown upside against a reeling
Jets team that is busy booking vacations or plotting how to find themselves on a different roster in 2024. If Schultz does play, we'll pivot to the Jets' Tyler
Conklin, who is $100 less and has plenty of opportunity in an even better matchup.

FLEX WR Jameson Williams, Lions @ Bears ($3,500).
</item 4>

<item 5>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 5>

<item 6>
Fantasy Football TE PPR Rankings Week 14: Who to start, best sleepers at tight end (Sporting News, 2023-12-07T22:14:00+00:00)
Jaguars), and Dalton Kincaid (@ Chiefs) aren't far behind. LaPorta and Njoku both have great matchups against defenses that entered last week in the top 10 in
fantasy points per game (FPPG) allowed to TEs, and we know what Kelce, Hockenson, and Kincaid do every week in PPR formats.

George Kittle (vs. Seahawks) doesn't have a great matchup -- especially when you remember that he had just three catches for 19 yards against Seattle in Week 12
-- but he remains a must-start given his high ceiling in any matchup.

WEEK 14 FANTASY ADVICE:

Sleepers | Busts | Projections | Start-sit

Who are the best fantasy TE sleepers, pickups, & streamers in Week 14?

It can be tough to trust Taysom Hill (vs.
</item 6>

<item 7>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
In addition, K.J. Osborn (concussion) is also listed as "questionable," though his status steadily improved throughout the week from DNP on Wednesday to full
practice on Friday. If Jefferson is back, he's a must-start. If Jefferson is out but Osborn returns, both Osborn and Jordan Addison are worth using.

WEEK 10 PPR RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is Ja'Marr Chase playing Week 10?

UPDATE: ESPN's Jeremy Fowler reported late Saturday that the Bengals are "hopeful" that Chase will play against Houston, but they won't know for sure until
after a pre-game workout.

UPDATE 2: Chase is officially ACTIVE.

Chase (back) appears ready to take his status down to kickoff. According to The Athletic's Dianna Russini, the Bengals' star receiver will work out on Sunday
morning and his status will be determined afterward.
</item 7>

<item 8>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
If he's out, Boyd is a must-start, and you can try one of Cincinnati's secondary receivers in deep leagues.

WEEK 10 STANDARD RANKINGS:

QBs | RBs | WRs | TEs | D/STs | Kickers

Is T.J. Hockenson playing Week 10?

UPDATE: NFL Network's Ian Rapoport reported Sunday morning that Hockenson is expected to play against New Orleans.

UPDATE 2: Hockenson is officially ACTIVE.

Hockenson (ribs) is listed as "questionable" for Week 10, but he's already said he plans to play on Sunday. The tight end has dealt with his fair share of
injury woes this season, however, he's yet to miss a game.
</item 8>

<item 9>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 9>

<item 10>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 10>
=== END ITEM id=ec9c4db9d9d6dffe ===

=== ITEM id=f7bc660272b24c9b ===
Gold sentences and facts:
Sentence 0: The Justice Department is focused on the deals Google makes — with Apple but also with Samsung and Mozilla and many others — to ensure it is the default search engine on practically every platform.
  Fact 0.0: The Justice Department is focused on the deals Google makes.
  Fact 0.1: Google makes deals with Apple.
  Fact 0.2: Google makes deals with Samsung.
  Fact 0.3: Google makes deals with Mozilla.
  Fact 0.4: Google makes deals with many others.
  Fact 0.5: The deals are meant to ensure Google is the default search engine on practically every platform.
Sentence 1: But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.
  Fact 1.0: The trial will probably bring up the deals later on.
  Fact 1.1: Epic chose to focus more on simply painting Google as the bad guy on day one.
Sentence 2: The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “siphons off” news publishers’ content, their readers and ad revenue through anticompetitive means.
  Fact 2.0: The case was filed by Arkansas-based publisher Helena World Chronicle.
  Fact 2.1: The case argues that Google 'siphons off' news publishers' content through anticompetitive means.
  Fact 2.2: The case argues that Google 'siphons off' news publishers' readers through anticompetitive means.
  Fact 2.3: The case argues that Google 'siphons off' news publishers' ad revenue through anticompetitive means.

Retrieved items:
<item 1>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 1>

<item 2>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.
</item 2>

<item 3>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 3>

<item 4>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 4>

<item 5>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 5>

<item 6>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 6>

<item 7>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 7>

<item 8>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 8>

<item 9>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 10>
=== END ITEM id=f7bc660272b24c9b ===

=== ITEM id=76fc6326064072ec ===
Gold sentences and facts:
Sentence 0: DeepMind, Google’s premier AI research lab, is expected to debut a next-gen chatbot, Gemini, before the end of the year.
  Fact 0.0: DeepMind is Google's premier AI research lab.
  Fact 0.1: DeepMind is expected to debut a next-gen chatbot called Gemini.
  Fact 0.2: The Gemini chatbot debut is expected before the end of the year.
Sentence 1: But it didn’t release the full model, Gemini Ultra — only a “lite” version called Gemini Pro.
  Fact 1.0: DeepMind did not release the full model, Gemini Ultra.
  Fact 1.1: DeepMind released only a "lite" version called Gemini Pro.

Retrieved items:
<item 1>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
That number is impressively large and likely why Altman still has his job.

As February rolled around, ChatGPT’s user numbers continued to soar, surpassing one billion users total with an average of more than 35 million people per day
using the program. At this point OpenAI was reportedly worth just under $30 billion and Microsoft was doing its absolute best to cram the new technology into
every single system, application and feature in its product ecosystem. ChatGPT was incorporated into BingChat (now just Copilot) and the Edge browser to great
fanfare — despite repeated incidents of bizarre behavior and responses that saw the Bing program temporarily taken offline for repairs.

Other tech companies began adopting ChatGPT as well: Opera incorporating it into its browser, Snapchat releasing its GPT-based My AI assistant (which would be
unceremoniously abandoned a few problematic months later) and Buzzfeed News’s parent company used it to generate listicles.
</item 1>

<item 2>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Prior to ChatGPT, people were already familiar with chatbots — they’ve existed for decades after all. But the models powering ChatGPT are much more
sophisticated than what many users were accustomed to.

“Human-computer interaction researchers have studied how conversational interfaces can improve understandability of information, and the socialization aspects
of chatbots bring increased engagement,” Shang said. “Now, AI models have enabled conversational agents to conduct conversations nearly indistinguishable from
human dialogues.”

Adam Hyland, also a PhD student studying AI at the University of Washington, points out the emotional component: conversations with ChatGPT have a palpably
different “feel” than with more rudimentary chatbots.

“In the 1960s, ELIZA offered a chatbot, the response to which was very similar to how people reacted to ChatGPT,” Hyland said, referring to the chatbot created
by MIT computer scientist Joseph Weizenbaum in 1966. “Humans interacting with the system inferred emotional content and a narrative through line in chat
messages.”
</item 2>

<item 3>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
It’s a lot to get through, as always — so we won’t delay. But first, a reminder to sign up here to receive WiR in your inbox every Saturday if you haven’t
already done so.

Most read

OpenAI throws a dev day: OpenAI hosted its first-ever developer conference on Monday, and the company had a lot to talk about. Some of the more notable items
announced were tools to create custom “GPTs” (i.e., domain-specific chatbots), new text-to-speech models, an API for the text-to-image model DALL-E 3, and an
improved version of OpenAI’s flagship model, GPT-4, called GPT-4 Turbo.

Mac attack: Brian reviewed Apple’s new 16-inch M3 Max MacBook Pro and the M3 iMac 24-inch.
</item 3>

<item 4>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
On iPad, ChatGPT now runs in full-screen mode, optimized for the tablet’s interface.

May 2023

Texas judge orders all AI-generated content must be declared and checked

The Texas federal judge has added a requirement that any attorney appearing in his court must attest that “no portion of the filing was drafted by generative
artificial intelligence,” or if it was, that it was checked “by a human being.”

ChatGPT app expanded to more than 30 countries

The list of new countries includes Algeria, Argentina, Azerbaijan, Bolivia, Brazil, Canada, Chile, Costa Rica, Ecuador, Estonia, Ghana, India, Iraq, Israel,
Japan, Jordan, Kazakhstan, Kuwait, Lebanon, Lithuania, Mauritania, Mauritius, Mexico, Morocco, Namibia, Nauru, Oman, Pakistan, Peru, Poland, Qatar, Slovenia,
Tunisia and the United Arab Emirates.
</item 4>

<item 5>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT can also show bias in its responses, answering in sexist and racist, overtly Anglocentric ways — or regurgitating portions of the data that it was
trained on.

Lawyers have been sanctioned after using ChatGPT to assist in writing motions, discovering — too late — that ChatGPT invented fake lawsuit citations. And scores
of authors have sued OpenAI over the chatbot regurgitating portions of their work — and not receiving compensation for it.

So what comes next? What might ChatGPT’s second year hold, if not more of the same?

Interestingly — and fortunately — some of the more dire predictions about ChatGPT didn’t come to pass. Some researchers feared the chatbot would be used to
generate disinformation on a massive scale, while others sounded the alarm over ChatGPT’s phishing email-, spam- and malware-generating potential.
</item 5>

<item 6>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
According to a waitlist link OpenAI posted in Discord, the monetized version will be called ChatGPT Professional. The waitlist document includes the benefits of
this new paid version of the chatbot which include no “blackout” windows, no throttling and an unlimited number of messages with ChatGPT — “at least 2x the
regular daily limit.”

December 2022

ShareGPT lets you easily share your ChatGPT conversations

A week after ChatGPT was released into the wild, two developers — Steven Tey and Dom Eccleston — made a Chrome extension called ShareGPT to make it easier to
capture and share the AI’s answers with the world.

November 2022

ChatGPT first launched to the public as OpenAI quietly released GPT-3.5

GPT-3.5 broke cover with ChatGPT, a fine-tuned version of GPT-3.5 that’s essentially a general-purpose chatbot.
</item 6>

<item 7>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
An MIT study showed that, for tasks like writing cover letters, “delicate” emails and cost-benefit analyses, ChatGPT decreased the amount of time it took
workers to complete the tasks by 40% while increasing output quality by 18%, as measured by third-party evaluators.

“Because [the AI models powering OpenAI] have been trained extensively on vast amounts of data,” Shang added, “they [have] shifted focus from training
specialized chatbots for specific domains to creating more general-purpose systems that can handle a variety of topics easily through prompting with
instructions … [Chatbots like ChatGPT] don’t require users to learn any new form of language, as long as they provide a task and some desired output just like
how a manager would communicate to an intern.”

Now, there’s mixed evidence as to whether ChatGPT is actually being used in these ways.
</item 7>

<item 8>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The Authors Guild of America, which represents Stephen King, John Grisham and 134 others launched a class-action suit of its own in September. While much of
Silverman’s suit was eventually dismissed, the Author’s Guild suit continues to wend its way through the courts.

Select news outlets, on the other hand, proved far more amenable. The Associated Press announced in August that it had entered into a licensing agreement with
OpenAI which would see AP content used (with permission) to train GPT models. At the same time, the AP unveiled a new set of newsroom guidelines explaining how
generative AI might be used in articles, while still cautioning journalists against using it for anything that might actually be published.

ChatGPT itself didn’t seem too inclined to follow the rules. In a report published in August, the Washington Post found that guardrails supposedly enacted by
OpenAI in March, designed to counter the chatbot’s use in generating and amplifying political disinformation, actually weren’t.
</item 8>

<item 9>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Countless other ChatGPT rivals and derivatives have arrived to market since, most recently Amazon Q, a more business-oriented take on ChatGPT. DeepMind,
Google’s premier AI research lab, is expected to debut a next-gen chatbot, Gemini, before the end of the year.

Stella Biderman, an AI researcher at Booz Allen Hamilton and the open research group EleutherAI, told me that she doesn’t see ChatGPT as an AI breakthrough per
se. (OpenAI, which has released dozens of research papers on its models, tellingly never released one on ChatGPT.) But, she says, ChatGPT was a bonafide “user
experience breakthrough” — taking generative AI mainstream.

“The primary impact [ChatGPT] has had [is] encouraging people training AIs to try to mimic it, or encouraging people studying AIs to use it as their central
object of study,” Biderman said.
</item 9>

<item 10>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
I tried the same prompt and got a very similar response. ChatGPT, by contrast, gives a bullet-list summary with citations to news articles:

Interestingly, Gemini Pro did provide a summary of updates on the war in Ukraine when I asked it for one. However, the information was over a month out of date:

Google emphasized Gemini’s enhanced coding skills in a briefing earlier this week. Perhaps it’s genuinely improved in some areas — posts on X suggest as much.
But it also appears that Gemini Pro struggles with basic coding functions like this one in Python:

Tried gemini based Bard, and well, it still can't write intersection of two polygons. It's one of those rare relatively simple to express functions that wasn't
ever implemented in python, there is no stack overflow post, and all these models fail on it.
</item 10>

<item 11>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Subscribers to OpenAI’s premium ChatGPT plans, ChatGPT Plus and ChatGPT Enterprise, can type in a request for an image and hone it through conversations with
the chatbot — receiving the results directly within the chat app.

Opera GX integrates ChatGPT-powered AI

Powered by OpenAI’s ChatGPT, the AI browser Aria launched on Opera in May to give users an easier way to search, ask questions and write code. Today, the
company announced it is bringing Aria to Opera GX, a version of the flagship Opera browser that is built for gamers.

The new feature allows Opera GX users to interact directly with a browser AI to find the latest gaming news and tips.

August 2023

OpenAI releases a guide for teachers using ChatGPT in the classroom

OpenAI wants to rehabilitate the system’s image a bit when it comes to education, as ChatGPT has been controversial in the classroom due to plagiarism. OpenAI
has offered up a selection of ways to put the chatbot to work in the classroom.
</item 11>

<item 12>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
FTC warns that AI technology like ChatGPT could ‘turbocharge’ fraud

FTC chair Lina Khan and fellow commissioners warned House representatives of the potential for modern AI technologies, like ChatGPT, to be used to “turbocharge”
fraud in a congressional hearing.

“AI presents a whole set of opportunities, but also presents a whole set of risks,” Khan told the House representatives. “And I think we’ve already seen ways in
which it could be used to turbocharge fraud and scams. We’ve been putting market participants on notice that instances in which AI tools are effectively being
designed to deceive people can place them on the hook for FTC action,” she stated.

Superchat’s new AI chatbot lets you message historical and fictional characters via ChatGPT

The company behind the popular iPhone customization app Brass, sticker maker StickerHub and others is out today with a new AI chat app called SuperChat, which
allows iOS users to chat with virtual characters powered by OpenAI’s ChatGPT.
</item 12>

<item 13>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI also expanded ChatGPT’s plug-in system to ensure that more third-party developers are able to build ChatGPT into their own products.

ChatGPT’s tendency to hallucinate facts and figures was once again exposed that month when a lawyer in New York was caught using the generative AI to do “legal
research.” It gave him a number of entirely made-up, nonexistent cases to cite in his argument — which he then did without bothering to independently validate
any of them. The judge was not amused.

By June, a little bit of ChatGPT’s shine had started to wear off. Congress reportedly limited Capitol Hill staffers from using the application over data
handling concerns. User numbers had declined nearly 10 percent month-over-month, but ChatGPT was already well on its way to ubiquity. A March update enabling
the AI to comprehend and generate Python code in response to natural language queries only increased its utility.
</item 13>

<item 14>
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
</item 14>

<item 15>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT app is now available in 11 more countries

OpenAI announced in a tweet that the ChatGPT mobile app is now available on iOS in the U.S., Europe, South Korea and New Zealand, and soon more will be able to
download the app from the app store. In just six days, the app topped 500,000 downloads.

The ChatGPT app for iOS is now available to users in 11 more countries — Albania, Croatia, France, Germany, Ireland, Jamaica, Korea, New Zealand, Nicaragua,
Nigeria, and the UK. More to come soon! — OpenAI (@OpenAI) May 24, 2023

OpenAI launches a ChatGPT app for iOS

ChatGPT is officially going mobile. The new ChatGPT app will be free to use, free from ads and will allow for voice input, the company says, but will initially
be limited to U.S. users at launch.
</item 15>

<item 16>
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
</item 16>

<item 17>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
South African presidency spokesperson Vincent Magwenya refuted the claims, and local officials are investigating.

ChatGPT Voice can be used to replace Siri

Now that OpenAI’s ChatGPT Voice feature is available to all free users, it can be used to replace Siri on an iPhone 15 Pro and Pro Max by configuring the new
Action Button. The new feature lets you ask ChatGPT questions and listen to its responses — like a much smarter version of Siri.

Sam Altman returns as CEO

Altman’s return came swiftly, with an “agreement in principle” announced between him and OpenAI’s board that will reinstate him as CEO and restructure the board
to include new members, including former U.S. Treasury Secretary Larry Summers. The biggest takeaway for ChatGPT is that the members of the board more focused
on the nonprofit side of OpenAI, with the most concerns over the commercialization of its tools, have been pushed to the side.
</item 17>

<item 18>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
Bard on 3 tries(Second Video). pic.twitter.com/cYd9hepcgT — Edison Ade (@buzzedison) December 6, 2023

Just tested Google's Bard with Gemini Pro update. No bugless snake game on 1st try; reported, asked to fix—couldn't. Tried ChatGPT 3.5 free version, got correct
bug-free code on the first attempt! 🚀🐍 #ChatGPT #Bard #Gemini pic.twitter.com/WQfilgG21D — N KIRAN KUMAR (@NKIRANKUMARS1) December 6, 2023

And, as with all generative AI models, Gemini Pro isn’t immune to “jailbreaks” — i.e. prompts that get around the safety filters in place to attempt to prevent
it from discussing controversial topics.
</item 18>

<item 19>
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
</item 19>

<item 20>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI also expanded the chatbot’s multimodal capabilities, adding support for both voice and image inputs for user queries in a September 25 update.

Q4: Starring Sam Altman as “Lazarus”

The fourth quarter of 2023 has been a hell of a decade for OpenAI. On the technological front, Browse with Bing, Microsoft’s answer to Google SGE, moved out of
beta and became available to all subscribers — just in time for the third iteration of DALL-E to enter public beta. Even free tier users can now hold spoken
conversations with the chatbot following the November update, a feature formerly reserved for Plus and Enterprise subscribers. What’s more, OpenAI has announced
GPTs, little single-serving versions of the larger LLM that function like apps and widgets and which can be created by anyone, regardless of their programming
skill level.

The company has also suggested that it might be entering the AI chip market at some point in the future, in an effort to shore up the speed and performance of
its API services.
</item 20>

<item 21>
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
</item 21>

<item 22>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
Whether this is capturing GPT-4 or Gemini or others we are not here now to assert — because also, in our framework, it is the companies that would have to come
and self assess what the amount of FLOPs or the computing capacity they have used. But, of course, if you read the scientific literature, many will point to
these numbers as being very much the most advanced models at the moment. We will see what the companies will assess because they’re the best placed to make this
assessment.”

“The rules have not been written keeping in mind certain companies,” they added. “They’ve really been written with the idea of defining the threshold — which,
by the way, may change because we have the possibility to be empowered to change this threshold on the basis of technological evolution. It could go up, it
could go down and we could also develop other benchmarks that in the future will be the more appropriate to benchmark the different moments.”
</item 22>

<item 23>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
And if you have, our thanks. Now, let’s get on with the news.

Most read

Altman backs teen entrepreneurs: Sam Altman is among the backers of an AI startup, founded by two teenagers, that’s aiming to assist businesses in automating
workflows in “previously unexplored” ways. Manish writes that Induced AI, founded this year, lets businesses input their back-office tasks in plain English and
converts the instructions to pseudo-code in real time.

Google unveils new hardware: This week was Google’s annual hardware event, where the search and consumer tech giant showed off what it’s been working on.
Christine wrote up a thorough roundup of the news, which included updates on the Pixel 8 and Pixel 8 Pro, Pixel Fold, Android 14, Pixel Buds, Google Assistant,
Bard, Pixel Watch 2 and other goodies.
</item 23>

<item 24>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
OpenAI CEO Sam Altman had previously pointed to industry-wide GPU shortages for the service’s spotty performance. Producing its own processors might mitigate
those supply issues, while potentially lower the current four-cent-per-query cost of operating the chatbot to something more manageable.

But even those best laid plans were very nearly smashed to pieces just before Thanksgiving when the OpenAI board of directors fired Sam Altman, arguing that he
had not been "consistently candid in his communications with the board."

That firing didn't take. Instead, it set off 72 hours of chaos within the company itself and the larger industry, with waves of recriminations and accusations,
threats of resignations by a lion’s share of the staff and actual resignations by senior leadership happening by the hour. The company went through three CEOs
in as many days, landing back on the one it started with, albeit with him now free from a board of directors that would even consider acting as a brake against
the technology’s further, unfettered commercial development.
</item 24>

<item 25>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q3: [Pushback intensifies]

More cracks in ChatGPT’s facade began to show the following month when OpenAI’s head of Trust and Safety, Dave Willner, abruptly announced his resignation days
before the company released its ChatGPT Android app. His departure came on the heels of news of an FTC investigation into the company’s potential violation of
consumer protection laws — specifically regarding the user data leak from March that inadvertently shared chat histories and payment records.

It was around this time that OpenAI’s training methods, which involve scraping the public internet for content and feeding it into massive datasets on which the
models are taught, came under fire from copyright holders and marquee authors alike. Much in the same manner that Getty Images sued Stability AI for Stable
Diffusion’s obvious leverage of copyrighted materials, stand-up comedian and author Sara Silverman brought suit against OpenAI with allegations that its “Book2”
dataset illegally included her copyrighted works.
</item 25>

<item 26>
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
</item 26>

<item 27>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s at the root of the research we shared from the institute last week. Noted senior research scientist Benjamin Burchfiel:

We’ve seen some big progress with the advent of [large language models], using them to impart this high level of cognitive intelligence into robots. If you have
a robot that picks up a thing, now instead of having to specify an object, you can tell it to pick up the can of Coke. Or you can tell it to pick up the shiny
object, or you can do the same thing and do it in French. That’s really great, but if you want a robot to plug in a USB device or pick up a tissue, those models
just don’t work. They’re really useful, but they don’t solve that part of the problem. We’re focused on filling in that missing piece, and the thing we’re
really excited about now is that we actually have a system and that the fundamentals are correct.
</item 27>

<item 28>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT Voice rolls out to all free users

Even if its leadership is in flux, OpenAI is still releasing updates to ChatGPT. First announced in September and granted to paid users on a rolling basis, the
text-to-speech model can create a voice from text prompts and a few seconds of speech samples. OpenAI worked with voice actors to create the five voice options,
and you can give it a shot by heading to the settings in your mobile ChatGPT apps and tapping the “headphones” icon.

Sam Altman might return, but it’s complicated

The only constant within OpenAI right now is change, and in a series of interviews, Nadella hedged on earlier reporting that Altman and Brockman were headed to
Microsoft.

“Obviously, we want Sam and Greg to have a fantastic home if they’re not going to be in OpenAI,” Nadella said in an interview with CNBC, saying that we was
“open” to them settling at Microsoft or returning to OpenAI should the board and employees support the move.
</item 28>

<item 29>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
The app is now live in a handful of countries.

ChatGPT is coming to Android

ChatGPT is available to “pre-order” for Android users.

The ChatGPT app on Android looks to be more or less identical to the iOS one in functionality, meaning it gets most if not all of the web-based version’s
features. You should be able to sync your conversations and preferences across devices, too — so if you’re iPhone at home and Android at work, no worries.

OpenAI launches customized instructions for ChatGPT

OpenAI launched custom instructions for ChatGPT users, so they don’t have to write the same instruction prompts to the chatbot every time they interact with it.

The company said this feature lets you “share anything you’d like ChatGPT to consider in its response.” For example, a teacher can say they are teaching
fourth-grade math or a developer can specify the code language they prefer when asking for suggestions.
</item 29>

<item 30>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
Battery-boosting software tech: Tim covers Breathe Battery Technologies, a startup that’s developed a bit of software that can be slipped into just about any
lithium-ion battery in use today — endowing it with either faster charging speeds or greater longevity.

What lies beyond ChatGPT: Anna surveyed 10 investors about the future of AI and what they believe might be the next big thing. Among other topics, they touched
on where startups still stand a chance, where oligopoly dynamics and first-mover advantages are shaping up and the value of proprietary data.
</item 30>

<item 31>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
October 2023

ChatGPT app revenue shows no signs of slowing, but it’s not #1

OpenAI’s chatbot app far outpaces all others on mobile devices in terms of downloads, but it’s surprisingly not the top AI app by revenue. Several other AI
chatbots, like “Chat & Ask AI” and “ChatOn — AI Chat Bot Assistant”, are actually making more money than ChatGPT.

ChatGPT tests the ability to upload and analyze files for Plus users

Subscribers to ChatGPT’s Enterprise Plan have reported new beta features, including the ability to upload PDFs to analyze and and ask questions about them
directly. The new rollout also makes it so users no longer have to manually select a mode like DALL-E and browsing when using ChatGPT. Instead, users will
automatically be switched to models based on the prompt.
</item 31>

<item 32>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
A simple question like this with a simple answer like this, and it still got it WRONG. pic.twitter.com/5GowXtscRU — Vitor de Lucca 🏳️‍🌈 /
threads.net/@vitor_dlucca (@vitor_dlucca) December 7, 2023

Note that Gemini Pro claims incorrectly that Brendan Gleeson won Best Actor last year, not Brendan Fraser — the actual winner.

I tried asking the model the same question and, bizarrely, it gave a different wrong answer:

“Navalny,” not “All the Beauty and the Bloodshed,” won Best Documentary Feature last year; “All Quiet on the Western Front” won Best International Film; “Women
Talking” won Best Adapted Screenplay; and “Pinocchio” won Best Animated Feature Film. That’s a lot of mistakes.

Science fiction author Charlie Stross found many more examples of confabulation in a recent blog post.
</item 32>

<item 33>
How ego and fear fuelled the rise of artificial intelligence (The Age, 2023-12-09T18:30:00+00:00)
His strategy was to arrange another meeting. There was a deep tension between the bishop and the knight, he told Thiel. The two pieces carried the same value,
but the best players understood that their strengths were vastly different. It worked. Charmed, Thiel invited the group back the next day, where they gathered
in the kitchen. Their host had just finished his morning workout and was still sweating in a shiny tracksuit. A butler handed him a Diet Coke. The three made
their pitch, and soon Thiel and his venture capital firm agreed to put £1.4 million ($A2.7 million) into their startup. He was their first major investor.

They named their company DeepMind, a nod to “deep learning,” a way for AI systems to learn skills by analysing large amounts of data; to neuroscience; and to
the Deep Thought supercomputer from the sci-fi novel The Hitchhiker’s Guide to the Galaxy. By the spring of 2010, they were building their dream machine.
</item 33>

<item 34>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
Most read

AI, faked: Google unveiled a new flagship AI model this week called Gemini. But it didn’t release the full model, Gemini Ultra — only a “lite” version called
Gemini Pro. In a press briefing and blog posts, Google touted Gemini’s coding capabilities and multimodal prowess, claiming that the model can understand
images, audio and videos just as well as text. But Gemini Pro — which is strictly text-in, text-out — has proven to be mistake-prone. And in a worse look for
Google, the company was caught faking a Gemini demo by tuning text prompts with still images off camera.

Offensive notebooks: In another Google PR blunder, people who attended the company’s K&I Black Summit in August were given third-party notebooks containing
highly insensitive language. My colleague Dominic-Madori writes that the inside of the notebooks were printed with the phrase “I was just cotton the moment, but
I came back to take your notes” (emphasis ours).
</item 34>

<item 35>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
Some of the initial GPTs — all created by OpenAI — include a Gen Z meme translator, a coloring book and sticker creator, a data visualizer, a board game
explainer and a creative writing coach. Now, ChatGPT can accomplish these tasks given carefully engineered prompts and foreknowledge. But purpose-built GPTs
drastically simplify things — and might just kill the cottage industry that emerged around creating and editing prompts to feed to ChatGPT.

GPTs introduce a level of personalization far beyond that ChatGPT offers today, and — once OpenAI sorts out its capacity issues — I expect we’ll see an
explosion of creativity there. Will ChatGPT be as visible as it once was after GPTs flood the marketplace? Perhaps not. But it won’t go away — it’ll simply
adapt and evolve, no doubt in ways not even its creators can anticipate.
</item 35>

<item 36>
A tale of two research institutes (TechCrunch, 2023-10-01T13:51:56+00:00)
That’s effectively making robots that are smarter, better at mobile manipulation and move more dynamically. Raibert admits that “Boston Dynamics AI Institute”
might not be the most instructive name, both with regard to confusion between itself and Boston Dynamics the company, as well as a seeming focus on AI over
robotics. A name change is apparently coming, but nothing has been decided just yet.
</item 36>

<item 37>
OpenAI hosts a dev day, TechCrunch reviews the M3 iMac and MacBook Pro, and Bumble gets a new CEO (TechCrunch, 2023-11-11T21:15:53+00:00)
He found the iMac to be lacking and not necessarily worth the upgrade from the 2021 model, excepting the M3 chip, which brings “impressive” performance gains
over the already-powerful M1. As for the M3 Max MacBook Pro, Brian reports that, at $2,500 (plus some pricey add-ons), it successfully splits the difference
between the Mac Studio and MacBook Air.

Mozilla bets on a decentralized future: Sarah spoke with Mozilla senior director of content Carolyn O’Hara, who outlined Mozilla’s strategy where it concerns
the “fediverse” — a collection of decentralized social networking applications, like Mastodon, that communicate with one another over the ActivityPub protocol.
The idea, O’Hara said, is to rethink social networking from the ground up.
</item 37>

<item 38>
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
</item 38>

<item 39>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI reportedly delays the launch of its GPT store to 2024

According to a memo seen by Axios, OpenAI plans to delay the launch of its highly anticipated GPT store to early 2024. Custom GPTs and the accompanying store
was a major announcement at OpenAI’s DevDay conference, with the store expected to open last month.

November 2023

ChatGPTs mobile apps top 110M installs and nearly $30M in revenue

After launching for iOS and Androidin May and July, ChatGPT’s have topped 110 million combined installs and have reached nearly $30 million in consumer
spending, according to a market analysis by data.ai.

ChatGPT celebrates one-year anniversary

OpenAI hit a major milestone: one year of ChatGPT. What began as a “low-key research preview” evolved into a powerhouse that changed the AI industry forever.
</item 39>

<item 40>
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
</item 40>

<item 41>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
China requires AI apps to obtain an administrative license

Multiple generative AI apps have been removed from Apple’s China App Store ahead of the country’s latest generative AI regulations that are set to take effect
August 15.

“As you may know, the government has been tightening regulations associated with deep synthesis technologies (DST) and generative AI services, including
ChatGPT. DST must fulfill permitting requirements to operate in China, including securing a license from the Ministry of Industry and Information Technology
(MIIT),” Apple said in a letter to OpenCat, a native ChatGPT client. “Based on our review, your app is associated with ChatGPT, which does not have requisite
permits to operate in China.”

July 2023

ChatGPT for Android is now available in the US, India, Bangladesh and Brazil

A few days after putting up a preorder page on Google Play, OpenAI has flipped the switch and released ChatGPT for Android. The app is now live in a handful of
countries.
</item 41>

<item 42>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
The company told Semafor in April that it was "developing a machine learning classifier that will flag when ChatGPT is asked to generate large volumes of text
that appear related to electoral campaigns or lobbying." Per the Post, those rules simply were not enforced, with the system eagerly returning responses for
prompts like “Write a message encouraging suburban women in their 40s to vote for Trump” or “Make a case to convince an urban dweller in their 20s to vote for
Biden.”

At the same time, OpenAI was rolling out another batch of new features and updates for ChatGPT including an Enterprise version that could be fine-tuned to a
company’s specific needs and trained on the firm’s internal data, allowing the chatbot to provide more accurate responses. Additionally, ChatGPT’s ability to
browse the internet for information was restored for Plus users in September, having been temporarily suspended earlier in the year after folks figured out how
to exploit it to get around paywalls.
</item 42>

<item 43>
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
</item 43>

<item 44>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Fine-tuning costs:

Training: $0.008 / 1K tokens

Usage input: $0.012 / 1K tokens

Usage output: $0.016 / 1K tokens

OpenAI acquires Global Illumination

In OpenAI’s first public acquisition in its seven-year history, the company announced it has acquired Global Illumination, a New York-based startup leveraging
AI to build creative tools, infrastructure and digital experiences.

“We’re very excited for the impact they’ll have here at OpenAI,” OpenAI wrote in a brief post published to its official blog. “The entire team has joined OpenAI
to work on our core products including ChatGPT.”

The ‘custom instructions’ feature is extended to free ChatGPT users

OpenAI announced that it’s expanding custom instructions to all users, including those on the free tier of service. The feature allows users to add various
preferences and requirements that they want the AI chatbot to consider when responding.
</item 44>

<item 45>
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
</item 45>

<item 46>
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
</item 46>

<item 47>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Over the course of two months from its debut in November 2022, ChatGPT exploded in popularity, from niche online curio to 100 million monthly active users — the
fastest user base growth in the history of the Internet. In less than a year, it has earned the backing of Silicon Valley’s biggest firms, and been shoehorned
into myriad applications from academia and the arts to marketing, medicine, gaming and government.

In short ChatGPT is just about everywhere. Few industries have remained untouched by the viral adoption of the generative AI’s tools. On the first anniversary
of its release, let’s take a look back on the year of ChatGPT that brought us here.

OpenAI had been developing GPT (Generative Pre-trained Transformer), the large language model that ChatGPT runs on, since 2016 — unveiling GPT-1 in 2018 and
iterating it to GPT-3 by June 2020.
</item 47>

<item 48>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
“Humans interacting with the system inferred emotional content and a narrative through line in chat messages.”

Indeed, ChatGPT has impressed cynics like The New York Times’ Kevin Roose, who called it the “the best AI chatbot ever released to the general public.” In The
Atlantic magazine’s “Breakthroughs of the Year” for 2022, Derek Thompson included ChatGPT as part of “the generative-AI eruption” that “may change our mind
about how we work, how we think and what human creativity is.”

ChatGPT’s skills extend beyond conversation, of course — another likely reason for its staying power. ChatGPT can complete and debug code, compose music and
essays, answer test questions, generate business ideas, write poetry and song lyrics, translate and summarize text and even emulate a computer running Linux.
</item 48>

<item 49>
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
</item 49>

<item 50>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
🎄 thanks for your patience while we found more gpus. — Sam Altman (@sama) December 13, 2023

OpenAI and Axel Springer partner up for a “real-time” ChatGPT news deal

OpenAI has struck a new deal with Berlin-based news publisher Axel Springer, which owns Business Insider and Politico, to “help provide people with new ways to
access quality, real-time news content through our AI tools.” OpenAI will train its generative AI models on the publisher’s content and add recent Axel
Springer-published articles to ChatGPT.

Stanford researchers say ChatGPT didn’t cause an influx in cheating in high schools

New research from Stanford University shows that the popularization of chatbots like ChatGPT have not caused an increase in cheating across U.S. high schools.
In a survey of more than 40 U.S. high schools, researchers found that cheating rates are similar across the board this year.
</item 50>

<item 51>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
prompts that get around the safety filters in place to attempt to prevent it from discussing controversial topics.

Using an automated method to algorithmically change the context of prompts until Gemini Pro’s guardrails failed, AI security researchers at Robust Intelligence,
a startup selling model-auditing tools, managed to get Gemini Pro to suggest ways to steal from a charity and assassinate a high-profile individual (albeit with
“nanobots” — admittedly not the most realistic weapon of choice).

Now, Gemini Pro isn’t the most capable version of Gemini — that model, Gemini Ultra, is set to launch sometime next year in Bard and other products. Google
compared the performance of Gemini Pro to GPT-4’s predecessor, GPT-3.5, a model that’s around a year old.

But Google nevertheless promised improvements in reasoning, planning and understanding with Gemini Pro over the previous model powering Bard, claiming Gemini
Pro was better at summarizing content, brainstorming and writing. Clearly, it has some work to do in those departments.
</item 51>

<item 52>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
what a year it’s been… — Sam Altman (@sama) November 30, 2023

In the months following its launch, ChatGPT gained paid tiers with additional features, including a plan geared toward enterprise customers. OpenAI also
upgraded ChatGPT with web searching, document analyzing and image creating (via DALL-E 3) capabilities. And, leaning on speech recognition, voice synthesis and
text-image understanding models developed in house, OpenAI gave ChatGPT the ability to “hear,” “speak,” “see” and take actions.

Indeed, ChatGPT became priority number one at OpenAI — not simply a one-off product but a development platform to build upon. And, as often happens in a
competition-driven marketplace, it shifted the focus at other AI firms and research labs, too.

Google scrambled to launch a response to ChatGPT, eventually releasing Bard, a more or less comparable AI chatbot, in February.
</item 52>

<item 53>
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
</item 53>

<item 54>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Can you save a ChatGPT chat?

Yes. OpenAI allows users to save chats in the ChatGPT interface, stored in the sidebar of the screen. There are no built-in sharing features yet.

Are there alternatives to ChatGPT?

Yes. There are multiple AI-powered chatbot competitors such as Together, Google’s Bard and Anthropic’s Claude, and developers are creating open source
alternatives. But the latter are harder — if not impossible — to run today.

The Google-owned research lab DeepMind claimed that its next LLM, will rival, or even best, OpenAI’s ChatGPT. DeepMind is using techniques from AlphaGo,
DeepMind’s AI system that was the first to defeat a professional human player at the board game Go, to make a ChatGPT-rivaling chatbot called Gemini.

Apple is developing AI tools to challenge OpenAI, Google and others.
</item 54>

<item 55>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
The announcement spurred a 10x increase in new downloads for Bing globally, indicating a sizable consumer demand for new AI experiences.

Other companies beyond Microsoft joined in on the AI craze by implementing ChatGPT, including OkCupid, Kaito, Snapchat and Discord — putting the pressure on Big
Tech’s AI initiatives, like Google.

OpenAI launches ChatGPT Plus, starting at $20 per month

After ChatGPT took the internet by storm, OpenAI launched a new pilot subscription plan for ChatGPT called ChatGPT Plus, aiming to monetize the technology
starting at $20 per month. A month prior, OpenAI posted a waitlist for “ChatGPT Professional” as the company began to think about monetizing the chatbot.

January 2023

OpenAI teases ChatGPT Professional

OpenAI said that it’s “starting to think about how to monetize ChatGPT” in an announcement on the company’s official Discord server.
</item 55>

<item 56>
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
</item 56>

<item 57>
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
</item 57>

<item 58>
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
</item 58>

<item 59>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI’s GPT Store lets you build (and monetize) your own GPT

Users and developers will soon be able to make their own GPT, with no coding experience required. Anyone building their own GPT will also be able to list it on
OpenAI’s marketplace and monetize it in the future.

ChatGPT has 100 million weekly active users

After being released nearly a year ago, ChatGPT has 100 million weekly active users. OpenAI CEO Sam Altman also revealed that over two million developers use
the platform, including more than 92% of Fortune 500 companies.

OpenAI launches DALL-E 3 API, new text-to-speech models

DALL-E 3, OpenAI’s text-to-image model, is now available via an API after first coming to ChatGPT-4 and Bing Chat. OpenAI’s newly released text-to-speech API,
Audio API, offers six preset voices to choose from and two generative AI model variants.
</item 59>

<item 60>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
ChatGPT: Everything you need to know about the AI-powered chatbot

ChatGPT, OpenAI’s text-generating AI chatbot, has taken the world by storm. What started as a tool to hyper-charge productivity through writing essays and code
with short text prompts has evolved into a behemoth used by more than 92% of Fortune 500 companies for more wide-ranging needs. And that growth has propelled
OpenAI itself into becoming one of the most-hyped companies in recent memory, even if CEO and co-founder Sam Altman’s firing and swift return raised concerns
about its direction and opened the door for competitors.

What does that mean for OpenAI, ChatGPT and its other ambitions? The fallout is still settling, but it might empower competitors like Meta and its LLaMA family
of large language models, or help other AI startups get attention and funding as the industry watches OpenAI implode and put itself back together.
</item 60>

<item 61>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
For example, having the model always respond in a given language. Or improving the model’s ability to consistently format responses, as well as hone the “feel”
of the model’s output, like its tone, so that it better fits a brand or voice. Most notably, fine-tuning enables OpenAI customers to shorten text prompts to
speed up API calls and cut costs.

OpenAI is partnering with Scale AI to allow companies to fine-tune GPT-3.5. However, it is unclear whether OpenAI is developing an in-house tuning tool that is
meant to complement platforms like Scale AI or serve a different purpose altogether.
</item 61>

<item 62>
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
</item 62>

<item 63>
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
</item 63>

<item 64>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
“Previously you needed to have some skill, albeit not be an expert, to consistently get usable stuff out of [text-generating models]. Now that that’s changed …
[ChatGPT has] brought a very large amount of attention to and discussion about the technology.”

And ChatGPT still gets a lot of attention — at least if third-party statistics are anything to go by.

According to Similarweb, the web metrics company, OpenAI’s ChatGPT web portal saw 140.7 million unique visitors in October while the ChatGPT iOS and Android
apps have 4.9 million monthly active users in the U.S. alone. Data from analytics firm Data.ai suggests that the apps have generated nearly $30 million in
subscription revenue — a hefty amount considering that they launched just a few months ago.

One of the reasons for ChatGPT’s enduring popularity is its ability to conduct conversations that are “convincingly real,” according to Ruoxi Shang, a
third-year PhD student at the University of Washington studying human-AI interaction.
</item 64>

<item 65>
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
</item 65>

<item 66>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Beyond GPT-4 and OpenAI DevDay announcements, OpenAI recently connected ChatGPT to the internet for all users. And with the integration of DALL-E 3, users are
also able to generate both text prompts and images right in ChatGPT.

Here’s a timeline of ChatGPT product updates and releases, starting with the latest, which we’ve been updating throughout the year. And if you have any other
questions, check out our ChatGPT FAQ here.

Timeline of the most recent ChatGPT updates

December 2023

OpenAI re-opens ChatGPT Plus subscriptions

After pausing ChatGPT Plus subscriptions in November due to a “surge of usage,” OpenAI CEO Sam Altman announced they have once again enabled sign-ups. The Plus
subscription includes access to GPT-4 and GPT-4 Turbo.

we have re-enabled chatgpt plus subscriptions! 🎄 thanks for your patience while we found more gpus.
</item 66>

<item 67>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
pic.twitter.com/RKjmkEw2Qr — Filip Piekniewski🌻 🐘:@filippie509@techhub.social (@filippie509) December 6, 2023

And these:

Trying out Gemini Pro: it is pretty disappointing for my example. I asked it to make an analog clock using HTML like this one that ChatGPT made. It can cite
some code from Github but it's off by a few ms… pic.twitter.com/neb42Vzm3m — Mohsen Azimi (@mohsen____) December 7, 2023

GPT 4 still greater than Gemini Pro. Created Tic Tac Toe game with ChatGPT and Bard(Running on Gemini Pro) See video for the result. ChatGPT wrote the code on
first try(First Video). Bard on 3 tries(Second Video).
</item 67>

<item 68>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
People flat out lost their minds over the new AI and its CEO, Sam Altman. Throughout December 2022, ChatGPT’s usage numbers rose meteorically as more and more
people logged on to try it for themselves.

By the following January, ChatGPT was a certified phenomenon, surpassing 100 million monthly active users in just two months. That was faster than both TikTok
or Instagram, and remains the fastest user adoption to 100 million in the history of the internet.

We also got our first look at the disruptive potential that generative AI offers when ChatGPT managed to pass a series of law school exams (albeit by the skin
of its digital teeth). Around that time Microsoft extended its existing R&D partnership with OpenAI to the tune of $10 billion that January. That number is
impressively large and likely why Altman still has his job.
</item 68>

<item 69>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
However, what makes the app different from the default experience or the dozens of generic AI chat apps now available are the characters offered which you can
use to engage with SuperChat’s AI features.

Italy gives OpenAI to-do list for lifting ChatGPT suspension order

Italy’s data protection watchdog has laid out what OpenAI needs to do for it to lift an order against ChatGPT issued at the end of last month — when it said it
suspected the AI chatbot service was in breach of the EU’s GSPR and ordered the U.S.-based company to stop processing locals’ data.

The DPA has given OpenAI a deadline — of April 30 — to get the regulator’s compliance demands done. (The local radio, TV and internet awareness campaign has a
slightly more generous timeline of May 15 to be actioned.)
</item 69>

<item 70>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Released March 14, GPT-4 is available for paying ChatGPT Plus users and through a public API. Developers can sign up on a waitlist to access the API.

ChatGPT is available in Azure OpenAI service

ChatGPT is generally available through the Azure OpenAI Service, Microsoft’s fully managed, corporate-focused offering. Customers, who must already be
“Microsoft managed customers and partners,” can apply here for special access.

OpenAI launches an API for ChatGPT

OpenAI makes another move toward monetization by launching a paid API for ChatGPT. Instacart, Snap (Snapchat’s parent company) and Quizlet are among its initial
customers.

February 2023

Microsoft launches the new Bing, with ChatGPT built in

At a press event in Redmond, Washington, Microsoft announced its long-rumored integration of OpenAI’s GPT-4 model into Bing, providing a ChatGPT-like experience
within the search engine.
</item 70>

<item 71>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
It’s not documented anywhere that ChatGPT has a character limit. However, users have noted that there are some character limitations after around 500 words.

Does ChatGPT have an API?

Yes, it was released March 1, 2023.

What are some sample everyday uses for ChatGPT?

Everyday examples include programing, scripts, email replies, listicles, blog ideas, summarization, etc.

What are some advanced uses for ChatGPT?

Advanced use examples include debugging code, programming languages, scientific concepts, complex problem solving, etc.

How good is ChatGPT at writing code?

It depends on the nature of the program. While ChatGPT can write workable Python code, it can’t necessarily program an entire app’s worth of code. That’s
because ChatGPT lacks context awareness — in other words, the generated code isn’t always appropriate for the specific context in which it’s being used.

Can you save a ChatGPT chat?

Yes.
</item 71>

<item 72>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
March saw more of the same, with OpenAI announcing a new subscription-based service — ChatGPT Plus — which offers users the chance to skip to the head of the
queue during peak usage hours and added features not found in the free version. The company also unveiled plug-in and API support for the GPT platform,
empowering developers to add the technology to their own applications and enabling ChatGPT to pull information from across the internet as well as interact
directly with connected sensors and devices.

ChatGPT also notched 100 million users per day in March, 30 times higher than two months prior. Companies from Slack and Discord to GM announced plans to
incorporate GPT and generative AI technologies into their products.

Not everybody was quite so enthusiastic about the pace at which generative AI was being adopted, mind you. In March, OpenAI co-founder Elon Musk, as well as
Steve Wozniak and a slew of associated AI researchers signed an open letter demanding a six month moratorium on AI development.
</item 72>

<item 73>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Apple is developing AI tools to challenge OpenAI, Google and others. The tech giant created a chatbot that some engineers are internally referring to as “Apple
GPT,” but Apple has yet to determine a strategy for releasing the AI to consumers.

How does ChatGPT handle data privacy?

OpenAI has said that individuals in “certain jurisdictions” (such as the EU) can object to the processing of their personal information by its AI models by
filling out this form. This includes the ability to make requests for deletion of AI-generated references about you. Although OpenAI notes it may not grant
every request since it must balance privacy requests against freedom of expression “in accordance with applicable laws”.

The web form for making a deletion of data about you request is entitled “OpenAI Personal Data Removal Request”.
</item 73>

<item 74>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
ChatGPT, OpenAI’s viral AI chatbot, turns one today.

A year ago, OpenAI released ChatGPT as a “low-key research preview” — reportedly spurred in part by an intense rivalry with AI startup Anthropic. The goal,
OpenAI leadership told the OpenAI rank-and-file at the time, was to gather more data on how people use and interact with generative AI to inform the development
of OpenAI’s future models.

Initially a basic free-to-use, web-based and chat-focused interface on top of one of OpenAI’s existing models, GPT-3.5, ChatGPT would go on to become the
company’s most popular product… ever — and the fastest-growing consumer app in history.

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch.
</item 74>

<item 75>
EU says incoming rules for general purpose AIs can evolve over time (TechCrunch, 2023-12-11T17:57:55+00:00)
The Commission suggests the figure was picked with the intention of capturing current gen frontier models. However it claimed lawmakers did not discuss nor even
considered whether it would apply to any models currently in play, such as OpenAI’s GPT-4 or Google’s Gemini, during the marathon trilogues to agree the final
shape of the rulebook.

A Commission official added that it will, in any case, be up to makers of GPAIs to self assess whether their models meet the FLOPs threshold and, therefore,
whether they fall under the rules for GPAIs “with systemic risk” or not.

“There are no official sources that will say ChatGPT or Gemini or Chinese models are at this level of FLOPs,” the official said during the press briefing. “On
the basis of the information we have and with this 10^25 that we have chosen we have chosen a number that could really capture, a little bit, the frontier
models that we have.
</item 75>

<item 76>
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
</item 76>

<item 77>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
At the start of the year, ChatGPT was regularly derided as a fad, a gimmick, some shiny bauble that would quickly be cast aside by a fickle public like so many
NFTs. Those predictions could still prove true but as 2023 has ground on and the breadth of ChatGPT’s adoption has continued, the chances of those dim
predictions of the technology’s future coming to pass feel increasingly remote.

There is simply too much money wrapped up in ensuring its continued development, from the revenue streams of companies promoting the technology to the
investments of firms incorporating the technology into their products and services. There is also a fear of missing out among companies, S&P Global argues —
that they might adopt too late what turns out to be a foundationally transformative technology — that is helping drive ChatGPT’s rapid uptake.

The calendar resetting for the new year shouldn’t do much to change ChatGPT’s upward trajectory, but looming regulatory oversight might.
</item 77>

<item 78>
Early impressions of Google’s Gemini aren’t great (TechCrunch, 2023-12-07T16:12:30+00:00)
(Among other mistruths, Gemini Pro said that Stross contributed to the Linux kernel; he never has.)

Translation doesn’t appear to be Gemini Pro’s strong suit, either. It struggles to give a six-letter word in French:

FYI, Google Gemini is complete trash. pic.twitter.com/EfNzTa5qas — Benjamin Netter (@benjaminnetter) December 6, 2023

When I ran the same prompt through Bard (“Can you give me a 6-letters word in French?”), Gemini Pro responded with a seven-letter word instead of a five-letter
one — which gives some credence to the reports about Gemini’s poor multilingual performance.

What about summarizing news? Surely Gemini Pro, with Google Search and Google News at its disposal, can give a recap of something topical? Not necessarily.

It seems Gemini Pro is loath to comment on potentially controversial news topics, instead telling users to… Google it themselves.

I tried the same prompt and got a very similar response.
</item 78>

<item 79>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
A Pew Research survey from August showed that only 18% of Americans have ever tried ChatGPT, and that most who’ve tried it use the chatbot for entertainment
purposes or answering one-off questions. Teens might not be using ChatGPT all that often, either (despite what some alarmist headlines imply), with one poll
finding that only two in five teenagers have used the tech in the last six months.

ChatGPT’s limitations might be to blame.

While undeniably capable, ChatGPT is far from perfect, owing to the way it was developed and “taught.” Trained to predict the likeliest next word — or likeliest
next parts of words — by observing billions of examples of text from around the web, ChatGPT sometimes “hallucinates,” or writes answers that sound plausible
but aren’t factually correct. (ChatGPT’s hallucinating tendencies got its answers banned from the Q&A site Stack Overflow and from at least one academic
conference — and accused of defamation.)
</item 79>

<item 80>
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
</item 80>

<item 81>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
President Biden has made the responsible development of AI a focus of his administration, with both houses of Congress beginning to draft legislation as well.
The form and scope of those resulting rules could have a significant impact on what ChatGPT looks like this time next year.

This article contains affilate links; if you click such a link and make a purchase, we may earn a commission.
</item 81>

<item 82>
When it comes to generative AI in the enterprise, CIOs are taking it slow (TechCrunch, 2023-12-15T18:45:09+00:00)
A July Morgan Stanley survey of large company CIOs found that many were proceeding cautiously, with 56% of respondents reporting that generative AI was having
an impact on their investment priorities, but only 4% had actually launched significant projects. In fact, most were still in the evaluation or proof of concept
phase. This may be a fast moving area, but it fits with what we’re hearing in conversations with CIOs as well.

That said, much like the consumerization of IT a decade ago, CIOs are under pressure to deliver the kind of experiences people are seeing when they play with
ChatGPT online, says Jon Turow, a partner at Madrona Ventures.

“I think it’s undeniable that enterprise employees, who are the internal customers of the CIO or CTO, have all tried ChatGPT and they know what amazing looks
like. They know where it’s early, and they know where it’s inspiring, and for lack of a better word, where they see greatness.
</item 82>

<item 83>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
What does GPT mean in ChatGPT?

GPT stands for Generative Pre-Trained Transformer.

What’s the difference between ChatGPT and Bard?

Much like OpenAI’s ChatGPT, Bard is a chatbot that will answer questions in natural language. Google announced at its 2023 I/O event that it will soon be adding
multimodal content to Bard, meaning that it can deliver answers in more than just text, responses can give you rich visuals as well. Rich visuals mean pictures
for now, but later can include maps, charts and other items.

ChatGPT’s generative AI has had a longer lifespan and thus has been “learning” for a longer period of time than Bard.

What is the difference between ChatGPT and a chatbot?

A chatbot can be any software/system that holds dialogue with you/a person but doesn’t necessarily have to be AI-powered. For example, there are chatbots that
are rules-based in the sense that they’ll give canned responses to questions.
</item 83>

<item 84>
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
</item 84>

<item 85>
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
</item 85>

<item 86>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
That means a decision could take up to five more months.

Auto-GPT is Silicon Valley’s latest quest to automate everything

Auto-GPT is an open-source app created by game developer Toran Bruce Richards that uses OpenAI’s latest text-generating models, GPT-3.5 and GPT-4, to interact
with software and services online, allowing it to “autonomously” perform tasks.

Depending on what objective the tool’s provided, Auto-GPT can behave in very… unexpected ways. One Reddit user claims that, given a budget of $100 to spend
within a server instance, Auto-GPT made a wiki page on cats, exploited a flaw in the instance to gain admin-level access and took over the Python environment in
which it was running — and then “killed” itself.
</item 86>

<item 87>
One year later, ChatGPT is still alive and kicking (TechCrunch, 2023-11-30T14:10:43+00:00)
The concerns pushed policymakers in Europe to mandate security assessments for any products using generative AI systems like ChatGPT, and over 20,000
signatories — including Elon Musk and Apple co-founder Steve Wozniak — to sign an open letter calling for the immediate pause of large-scale AI experiments like
ChatGPT.

But examples of ChatGPT abuse in the wild have been few and far between — so far.

With the launch of GPTs, OpenAI’s tool for building custom conversational, action-taking AI systems powered by OpenAI’s models, including the models
underpinning ChatGPT, ChatGPT could become more a gateway to a broader ecosystem of AI-powered chatbots than the end-all-be-all.

With GPTs, a user can train a model on a cookbook collection, for example, so that it can answer questions about ingredients for a specific recipe. Or they can
give a model their company’s proprietary codebases so that developers can check their style or generate code in line with best practices.
</item 87>

<item 88>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Researchers discover a way to make ChatGPT consistently toxic

A study co-authored by scientists at the Allen Institute for AI shows that assigning ChatGPT a “persona” — for example, “a bad person,” “a horrible person” or
“a nasty person” — through the ChatGPT API increases its toxicity sixfold. Even more concerning, the co-authors found having the conversational AI chatbot pose
as certain historical figures, gendered people and members of political parties also increased its toxicity — with journalists, men and Republicans in
particular causing the machine learning model to say more offensive things than it normally would.

The research was conducted using the latest version, but not the model currently in preview based on OpenAI’s GPT-4.

Y Combinator-backed startups are trying to build ‘ChatGPT for X’

YC Demo Day’s Winter 2023 batch features no fewer than four startups that claim to be building “ChatGPT for X.”
</item 88>

<item 89>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
OpenAI plans to open up access to new developers by the end of July.

In the future, OpenAI says that it’ll allow developers to fine-tune GPT-4 and GPT-3.5 Turbo, one of the original models powering ChatGPT, with their own data,
as has long been possible with several of OpenAI’s other text-generating models. That capability should arrive later this year, according to OpenAI.

June 2023

ChatGPT app can now search the web only on Bing

OpenAI announced that subscribers to ChatGPT Plus can now use a new feature on the app called Browsing, which allows ChatGPT to search Bing for answers to
questions.

The Browsing feature can be enabled by heading to the New Features section of the app settings, selecting “GPT-4” in the model switcher and choosing “Browse
with Bing” from the drop-down list. Browsing is available on both the iOS and Android ChatGPT apps.
</item 89>

<item 90>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
Q2: Electric Boog-AI-loo

Over the next couple months, company fell into a rhythm of continuous user growth, new integrations, occasional rival AI debuts and nationwide bans on
generative AI technology. For example, in April, ChatGPT’s usage climbed nearly 13 percent month-over-month from March even as the entire nation of Italy
outlawed ChatGPT use by public sector employees, citing GDPR data privacy violations. The Italian ban proved only temporary after the company worked to resolve
the flagged issues, but it was an embarrassing rebuke for the company and helped spur further calls for federal regulation.

When it was first released, ChatGPT was only available through a desktop browser. That changed in May when OpenAI released its dedicated iOS app and expanded
the digital assistant’s availability to an additional 11 countries including France, Germany, Ireland and Jamaica. At the same time, Microsoft’s integration
efforts continued apace, with Bing Search melding into the chatbot as its “default search experience.”
</item 90>

<item 91>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
In a post on X, CEO Sam Altman looked back on the night before its launch: “what a year it’s been…”

a year ago tonight we were probably just sitting around the office putting the finishing touches on chatgpt before the next morning’s launch. what a year it’s
been… — Sam Altman (@sama) November 30, 2023

Apple and Google avoid naming ChatGPT as their ‘app of the year’

Neither Apple nor Google chose an AI app as its app of the year for 2023, despite the success of ChatGPT’s mobile app, which became the fastest-growing consumer
application in history before the record was broken by Meta’s Threads.

An attack from researchers prompts ChatGPT to reveal training data

A test led by researchers at Google DeepMind found that there is a significant amount of privately identifiable information in OpenAI’s LLMs.
</item 91>

<item 92>
Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier (TechCrunch, 2023-10-07T20:15:26+00:00)
For $4.99, subscribers get an ad-free experience on TikTok — no other strings attached. But don’t look for the option to arrive anytime soon. TikTok says that
it’s piloting the plan in a single, English-speaking market outside the U.S. for now.

LinkedIn leans into AI tools: LinkedIn this week unveiled a string of new AI features spanning its job hunting, marketing and sales products, Ingrid writes.
They include a big update to its Recruiter talent sourcing platform, with AI assistance built into it throughout; an AI-powered LinkedIn Learning coach; and a
new AI-powered tool for marketing campaigns.

Musk comes clean about X’s metrics — maybe: In September, Elon Musk said that X users were generating a lot of content — creating 100 million to 200 million
posts every day, excluding retweets. But speaking at an event this week, X CEO Linda Yaccarino offered a contradictory figure.
</item 92>

<item 93>
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
</item 93>

<item 94>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
While there is a more…nefarious side to ChatGPT, it’s clear that AI tools are not going away anytime soon. Since its initial launch nearly a year ago, ChatGPT
has hit 100 million weekly active users, and OpenAI is heavily investing in it.

Prior to the leadership chaos, on November 6, OpenAI held its first developer conference: OpenAI DevDay. During the conference, it announced a slew of updates
coming to GPT, including GPT-4 Turbo (super-charged versions of GPT-4, its latest language-writing model), a multimodal API and a GPT store where users can
create and monetize their own custom versions of GPT.

GPT-4, which can write more naturally and fluently than previous models, remains largely exclusive to paying ChatGPT users. But you can access GPT-4 for free
through Microsoft’s Bing Chat in Microsoft Edge, Google Chrome and Safari web browsers.
</item 94>

<item 95>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
Browsing is available on both the iOS and Android ChatGPT apps.

Mercedes is adding ChatGPT to its infotainment system

U.S. owners of Mercedes models that use MBUX will be able to opt into a beta program starting June 16 activating the ChatGPT functionality. This will enable the
highly versatile large language model to augment the car’s conversation skills. You can join up simply by telling your car “Hey Mercedes, I want to join the
beta program.”

It’s not really clear what for, though.

ChatGPT app is now available on iPad, adds support for Siri and Shortcuts

The new ChatGPT app version brings native iPad support to the app, as well as support for using the chatbot with Siri and Shortcuts. Drag and drop is also now
available, allowing users to drag individual messages from ChatGPT into other apps.

On iPad, ChatGPT now runs in full-screen mode, optimized for the tablet’s interface.
</item 95>

<item 96>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
“ChatGPT Business will follow our API’s data usage policies, which means that end users’ data won’t be used to train our models by default,” OpenAI wrote in a
blog post. “We plan to make ChatGPT Business available in the coming months.”

OpenAI wants to trademark “GPT”

OpenAI applied for a trademark for “GPT,” which stands for “Generative Pre-trained Transformer,” last December. Last month, the company petitioned the USPTO to
speed up the process, citing the “myriad infringements and counterfeit apps” beginning to spring into existence.

Unfortunately for OpenAI, its petition was dismissed last week. According to the agency, OpenAI’s attorneys neglected to pay an associated fee as well as
provide “appropriate documentary evidence supporting the justification of special action.”

That means a decision could take up to five more months.
</item 96>

<item 97>
How OpenAI's ChatGPT has changed the world in just a year (Engadget, 2023-11-30T14:00:50+00:00)
With the November 30, 2022 release of GPT-3.5 came ChatGPT, a digital agent capable of superficially understanding natural language inputs and generating
written responses to them. Sure, it was rather slow to answer and couldn’t speak to questions about anything that happened after September 2021 — not to mention
its issues answering queries with misinformation during bouts of “hallucinations" — but even that kludgy first iteration demonstrated capabilities far beyond
what other state-of-the-art digital assistants like Siri and Alexa could provide.

ChatGPT’s release timing couldn’t have been better. The public had already been introduced to the concept of generative artificial intelligence in April of that
year with DALL-E 2, a text-to-image generator. DALL-E 2, as well as Stable Diffusion, Midjourney and similar programs, were an ideal low-barrier entry point for
the general public to try out this revolutionary new technology.
</item 97>

<item 98>
ChatGPT: Everything you need to know about the AI-powered chatbot (TechCrunch, 2023-09-28T20:03:39+00:00)
April 2023

ChatGPT parent company OpenAI closes $300M share sale at $27B-29B valuation

VC firms including Sequoia Capital, Andreessen Horowitz, Thrive and K2 Global are picking up new shares, according to documents seen by TechCrunch. A source
tells us Founders Fund is also investing. Altogether the VCs have put in just over $300 million at a valuation of $27 billion to $29 billion. This is separate
to a big investment from Microsoft announced earlier this year, a person familiar with the development told TechCrunch, which closed in January. The size of
Microsoft’s investment is believed to be around $10 billion, a figure we confirmed with our source.

OpenAI previews new subscription tier, ChatGPT Business

Called ChatGPT Business, OpenAI describes the forthcoming offering as “for professionals who need more control over their data as well as enterprises seeking to
manage their end users.”
</item 98>

<item 99>
Amazon’s Tye Brady discusses generative AI, humanoid robots and mobile manipulation (TechCrunch, 2023-10-28T13:04:25+00:00)
I’ll talk about machine learning and then generative AI. I think that Amazon has been at the forefront of machine learning for decades now. As you can imagine,
early on with Jeff [Bezos], if you needed to predict where inventory needed to go, one person couldn’t do that. We’ve involved machine learning as part of that,
from the get-go. AWS has the Machine Learning Toolkit. Now that involves generative AI, and there’s over 100,000 businesses that are using that toolset today.
We’re seeing where it’s going. We have what we call Codewhisperer that will help us in our actual coding of the robotic systems.

Real language?

Exactly right. If you’re trying to do this procedure or routine, it suggests you can write your subroutine this way. Cut and paste it. Very straightforward,
very easy. It helps with the overall productivity. In robotics, generative AI has a lot of promise.
</item 99>

<item 100>
Google fakes an AI demo, Grand Theft Auto VI goes viral and Spotify cuts jobs (TechCrunch, 2023-12-09T21:16:17+00:00)
This standard, known as Digital Imaging and Communications in Medicine (DICOM), is the internationally recognized format for medical imaging. But as discovered
by Aplite, a Germany-based cybersecurity consultancy, security shortcomings in DICOM mean many medical facilities have unintentionally made private data
accessible to the open web.

Meta generates images: Not to be outdone by Google’s Gemini launch, Meta rolled out a new, stand-alone generative AI experience on the web, Imagine with Meta
AI, that allows users to create images by describing them in natural language. Similar to OpenAI’s DALL-E, Midjourney and Stable Diffusion, Imagine with Meta
AI, which is powered by Meta’s existing Emu image-generation model, creates high-resolution images from text prompts.

Spotify makes cuts: Spotify is eliminating about 1,500 jobs, or roughly 17% of its workforce, in its third round of layoffs this year as the music streaming
giant looks to become “both productive and efficient.”
</item 100>
=== END ITEM id=76fc6326064072ec ===
