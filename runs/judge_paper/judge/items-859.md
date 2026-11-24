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

This file has 2 items.

=== ITEM id=98c1e244d9b54763 ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: He revealed that he made a friendship bracelet.
  Fact 0.1: The friendship bracelet is a fan tradition at Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: Making the friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: He later invited the 'Anti-Hero' singer on his podcast to one of his games at Arrowhead Stadium.
  Fact 1.1: The 'Anti-Hero' singer famously accepted the invitation on 24 September.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has now shared that she has nothing to hide.
  Fact 2.1: Swift has shared that she has nothing to hide in her relationship with Kelce.
  Fact 2.2: Swift's relationship with Kelce is new.

Retrieved items:
<item 1>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 1>

<item 2>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 3>

<item 4>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
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
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 6>

<item 7>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Well I would say that Travis Kelce's had a lot of big catches in his career. This would be the biggest."

Sept. 26: Chiefs games get even more popular

"Welcome to New York (*New Jersey), it's been waiting for you." - Welcome to New York

The defending Super Bowl champions are already one of the more popular teams in the league, but it is becoming the hottest ticket in town, no matter what town
that is.

"Since Swift's appearance at the Chiefs game on Sunday, the Jets have sold more tickets for their game against the Chiefs in a single day than since the season
started -- by more than double," a representatives at StubHub told to Town & Country magazine.

Sept. 27: Mahomes confirms he met Swift

"It was enchanting to meet you, all I can say is I was enchanted to meet you."
</item 7>

<item 8>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 8>

<item 9>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“I placed my hand on the gentleman’s back to let him know I was behind him. If I would have pushed him, he probably would have turned around and tased me,” he
said.

The Independent has contacted representatives for Taylor Swift and Travis Kelce for contact.
</item 9>

<item 10>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 10>

<item 11>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
We will see what happens in the near future."

Sept. 22: Another bracelet

"Lately [he's] been dressing for revenge." - Vigilante Sh--

Kelce was spotted at his nonprofit charity event wearing a bracelet that says "Vigilante Sh--," the name of one of Swift's songs.

Sept. 24: Swift attends Chiefs game

"Dating the boy on the football team." - Fifteen

Swift took Kelce up on his offer and attended the Chiefs' Week 3 game against the Chicago Bears. Swift was dressed on theme and was watching intently, cheering
and even chest bumping someone during big plays. She was especially thrilled to see Kelce get in the end zone, a three-yard catch that put the Chiefs up 41-0.
Kansas City went on to win 41-10 to improve to 2-1.

Sept. 24: Swift mingles with Donna Kelce



"Talkin' over a football team."
</item 11>

<item 12>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 12>

<item 13>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
The now-viral photo shows how Kelce may have hinted at the ongoing dating rumours before Swift attended his game. The snap showed Kelce posing at the Kelce Car
Jam charity event on Friday 22 September, where he was sporting a beaded friendship bracelet that read: “Vigilante S***”

The beads on the friendship bracelet appeared to reference Swift’s song “Vigilante S***” from her 2022 album, Midnights. In addition to the Swiftie-themed
jewellery, Kelce also wore a white striped shirt with silver chain necklaces, black sunglasses, and black pants for the event.

In the comments, fans both praised and questioned Kelce for the beaded friendship bracelet. Many people also pointed out how fans of the “Bad Blood” singer
famously trade and wear friendship bracelets at Swift’s Eras Tour concerts.
</item 13>

<item 14>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 14>

<item 15>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the episode of his podcast, the Philadelphia Eagles star asked his mother, Donna, which of her two sons she was “rooting for,” as the Eagles and Chiefs were
preparing to play against each other in the Super Bowl at the time.

“You’ve said in the past to both of us that you root for me because I’ve given you grandkids,” Jason, who shares three children – Wyatt, four, Elliotte, two,
and Bennett, seven months – with wife Kylie Kelce, said. “I’ve always known that you root for Travis. So who do you actually root for?”

After Jason asked his mother again who she was rooting for, “if in a battle to the death,” Donna responded by asking who he’d pick to save if she and her
husband, Ed Kelce, were stranded in the middle of the ocean.
</item 15>

<item 16>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
ADVERTISEMENT Article continues below this ad

How can you not love Taylor Swift? Most of the 12 Grammy Awardee’s songs are about that warm fuzzy feeling and chemical reaction that we call love. And her
songs have earned the support of 103.7992 million Swifties in the world. Yeah, we don’t have a round figure for that. We have the exact figure up to four
decimals. That’s how much data we have on her. Football has 184 million fans. The league is able to generate a revenue of $18 billion from 184 million. If it
adds 104 (after rounding 103.7992) million to that number, guess how much revenue it will generate? $28 billion. And just like that, their target will be met.
</item 16>

<item 17>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 17>

<item 18>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 18>

<item 19>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
This was the first, albeit tentative, confirmation from the media rather than from amateur Swiftie sleuths on social media. So, it was only a matter of time
before it entered the cultural zeitgeist. Blank Space Which it did, very quickly. On September 17, Kelce scored a touchdown against the Jaguars, triggering CBS
announcer Ian Eagle to reference Swift’s 2014 hit, Blank Space.

“Kelce finds a blank space for the score,” Eagle said, poking fun at Kelce’s newfound Swiftie status. And Eagle wasn’t the only one, with NFL Network’s Rich
Eisen including a slew of lyric references during a preshow, including “shake it off”, “anti-hero” and “all too well”. Kelce even acknowledged the
tongue-in-cheek banter, commenting on a montage video of the preshow that Eisen posted to Instagram. “Well played Rich … Well played,” he wrote. Suddenly,
football fans and Swifties were living through this surprising collision.
</item 19>

<item 20>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.

“Take away your feelings for Taylor,” Jason added. “What is your honest opinion on how the NFL is treating celebrities at the games?”

Seemingly beating around the bush, Kelce remarked: “I think it brings a little more to the atmosphere, it brings a little bit more to what you’re watching. But
at the same time, I think...”

Before his younger brother could continue his thought, the Philadelphia Eagles player chimed in: “They’re overdoing it.”

“They’re overdoing it a little bit, for sure,” Kelce agreed. “Especially my situation. I think they’re just trying to have fun with it.”

While sports lovers may not be too pleased with the NFL’s focus on Swift, the football league’s supposed obsession with ongoing Swift and Kelce dating rumours
has paid off in terms of ratings.
</item 20>

<item 21>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.
</item 21>

<item 22>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 22>

<item 23>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But before he could even answer, the answer was given for him.

Sept. 14: Rumors aren't denied

"The rumors are terrible and cruel, but honey, most of them are true." - New Romantics

Of course people were going to ask Jason, who is very close with his brother, whether rumors of the two hanging out were true. He did not confirm them, but he
also did not deny them.

Sept. 17: The references begin

"I watched it begin again." - Begin Again

The rumors began to pick up, and now there were a lot of people, commentators included, making Swift references around Kelce.

When the tight end caught a touchdown pass from Patrick Mahomes, CBS Sports' Ian Eagle said, "Kelce finds a blank space for the score."

This was only the start to many, many, Swift lyrics being referenced during NFL games.

Sept. 21: Kelce sends Swift an invite

"So inviting, I almost jump in."
</item 23>

<item 24>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

In addition to the kiss, the football star also had the perfect reaction to a fan calling him Swift’s boyfriend. After a fan shouted: “Let’s go Taylor’s
boyfriend” while Kelce was waiting for an elevator, he responded with a small smile on his face and by pumping his fist into the air.

Aside from their appearances at sports events, Swift and Kelce have also been photographed together on other occasions, including a date earlier this month in
New York City after their surprise appearances on the season 49 premiere of Saturday Night Live. At the time, sources alleged to Entertainment Tonight that,
during the date, Kelce got a little too close to Swift’s security personnel and allegedly told her bodyguard to step aside.

However, during an episode of his podcast, Kelce explained what he said to the security, and clarified that he didn’t push the bodyguard.
</item 24>

<item 25>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 25>

<item 26>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 26>

<item 27>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce’s previous comments about having children have sparked controversy amid his ongoing romance with Taylor Swift.

The Kansas City Chiefs player, 34, first made the remarks about starting his own family during a February 2023 episode of his and his brother Jason Kelce’s
podcast, New Heights. The episode resurfaced after a source claimed to Yahoo Entertainment that Swift and Travis Kelce “have talked about kids” within the next
year.
</item 27>

<item 28>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
87 with gusto.

The 12-time Grammy winner appeared to shout “let’s f—ing go” after Kelce scored a touchdown before chest-bumping other Chiefs fans. Considering Swift is known
for being an Eagles fan – she’s originally from Pennsylvania and even references the team in Gold Rush – her outward support for the Kelce’s team seemed extra
telling. Swifties were immediately on high alert, with some capturing Swift and Kelce leaving the stadium together. If that wasn’t enough, extra observant fans
were able to note that Kelce’s blue and white post-game outfit was supposedly labelled the “1989 bedroom painting set”, perhaps an intentional reference to
Swift’s 2014 album, which she will be re-releasing in October. Swift’s penchant for Easter eggs must be rubbing off on the tight end player. As if this weren’t
enough subtle references already, they were also spotted in a “getaway car” together outside the stadium – potentially a nod towards her 2017 track.
</item 28>

<item 29>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 29>

<item 30>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.
</item 30>

<item 31>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 31>

<item 32>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
</item 32>

<item 33>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Head coach Andy Reid was also asked about his star tight end's supposed relationship, to which he quipped, "I set them up."

Sept. 24: Kelce and Swift leave stadium together

"You know when it's time to go." - It's Time To Go

This was the first time the two were photographed together. They walked side by side, Kelce fittingly wearing an outfit with "1989" in the title and Swift
wearing a Chiefs zip-up.

Sept. 24: The two leave in a convertible

"Nothing good starts in the getaway car." - Getaway Car

Swift and Swift were spotted in a convertible, with the tight end driving, after the Chiefs' win. Kelce reportedly rented out a restaurant for a private party
with Swift and his teammates.

"Taylor Swift likes that Travis Kelce 'pursued her' and the two are having a 'great time getting to know each other,'" a source told Entertainment Tonight.
</item 33>

<item 34>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
The dating speculation first began in July, when Kelce confessed that he tried and failed to give Swift a friendship bracelet with his number on it during her
Eras Tour show. On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24
September. Since then, Swift has attended three more of Kelce’s games.

Kelce showed his support for Swift when he attended her Eras Tour show in Buenos Aires, Argentina, on 11 November. During the concert, Swift gave a shout-out to
the Chiefs player when she changed her “Karma” lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang,
letting out a laugh as fans in the audience screamed in response. The “All Too Well” singer was later seen running into his arms and kissing the NFL player
following the concert, as he whisked her away backstage.
</item 34>

<item 35>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
24: Swift mingles with Donna Kelce



"Talkin' over a football team." - Mean

Swift already seems to be making friends with the family. Swift was sitting next to Donna in the box and the two were seen talking and laughing as the game went
along.

Sept. 24: Postgame interviews turn into Swiftie talk

The postgame press conference seemed more about Swift and Kelce than it did about the actual game. The media wanted any insight it could get into the
relationship and how Swift ended up at Arrowhead.

"He told me at the last minute," Mahomes said when asked if he knew Swift was coming. "There's some things with Trav where he kind of just says it and you don't
know if it's true or not. He says it so calmly. ... Friday he was like, 'Yeah, I think she's coming to the game this weekend,' and then moved about his
business."
</item 35>

<item 36>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.
</item 36>

<item 37>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 37>

<item 38>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"I was disappointed that she doesn't talk before or after her shows because she has to save her voice for the 44 songs she sings ... She doesn't meet anybody,
or at least she didn't want to meet me, so I took it personal," Kelce said on an episode of the "New Heights" podcast he does with his brother, Philadelphia
Eagles center Jason Kelce.

He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in
Swift. When his brother asked if it was his jersey number or his phone number, Kelce replied, "you know which one."

"If you're up on Taylor Swift concerts, there are friendship bracelets and I had received a bunch of them, but I wanted to give Taylor Swift one with my number
on it."

Aug. 5: Kelce names his celebrity crush

Well before the two were rumored to be dating, Kelce was asked his crush.
</item 38>

<item 39>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
21: Kelce sends Swift an invite

"So inviting, I almost jump in." - Gold Rush

While on the Pat McAfee show, Kelce discussed the rumors and sounded somehow surprised people are talking about it.

"It is hilarious how much traction this has actually got," he said of all the talk. "It's like that old game in school called telephone where everybody is just
whispering in each others ear, just hearing some random stuff. Especially when you got Jason Kelce on live television telling people just both sides."

He then admitted that he invited Swift to one of his games, joking that they would compete to see who put on the bigger show at Arrowhead Stadium, home of the
Chiefs.

"I threw it out there, I threw the ball in her court and I told her you know, I've seen you rock the stage in Arrowhead you might have to come see me rock the
stage in Arrowhead and see which one is a little more lit. We will see what happens in the near future."
</item 39>

<item 40>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 40>

<item 41>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Could they have been driving off into the sunset? Of course, none of this actually confirms Swift and Kelce are actually dating. But regardless of whether it’s
a friendship or a romantic relationship, it has sent shockwaves through two massive cultural franchises, expanding the NFL’s audience into uncharted territory.
It’s a cultural pairing on par with – and perhaps even bigger than – Kim Kardashian and Kanye West.
</item 41>

<item 42>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
How do you not get that? That's called images right there."

Oct. 1: Swift attends Chiefs-Jets

"Welcome to New York, it's been waiting for you," - Welcome to New York

OK, so the Jets technically play in New Jersey, but we're just basing this off their name. Swift went to her second consecutive Chiefs game, showing up with
some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there were reports that the singer would be at the game.
The Chiefs defeated the Jets 23-20.

Oct. 5: NFL defends itself

Swift coverage is everywhere and the league even changed their social media bio to one of the singer's lyrics. The league's X (formerly Twitter) bio to read,
"NFL (Taylor's Version)," while their Instagram account read, "Chiefs are 2-0 as Swifties."
</item 42>

<item 43>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Yeah, my personal life -- that's not so personal. I did this to myself Jason, I know this. What's real is that you know it is my personal life. ... She's not
in the media as much as I am doing this show every single week."

Sept. 27: Jerry Jones comments on the Swift effect

Swift is getting the owners talking, too.

"It is hard to move that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a
lot of eye balls. She blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core
constituency. We all, I think, felt the same way when she's sitting up there with [Travis Kelce's] mom and she's going nuts when he's playing and those hearts
are popping out everywhere.
</item 43>

<item 44>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.
</item 44>

<item 45>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“OF ALL 230 SONGS IN THE DISCOGRAPHY I’M SICK,” one fan commented, while another person wrote: “It’s unbelievable how invested I am in this.”

“I’m gonna need to go back and see all photos of him from the last three months and dissect if he’s wearing friendship bracelets,” a third wrote.

In other photos from the charity event - which were shared to Instagram by Jordan’s KC photography - Kelce wasn’t seen wearing the friendship bracelet, leading
many people to believe that he received the bracelet from a fan that day.

Two days after his charity event, the “All Too Well” singer was spotted cheering for the Kansas City Chiefs tight end from the box seats at Arrowhead Stadium in
Kansas City, Missouri. Not only was she seen sitting next to Kelce’s mother, Donna, but she also appeared to yell “let’s f***ing go” in response to Kelce’s
third-quarter touchdown reception.
</item 45>

<item 46>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Unfortunately, the bracelet never made it to her wrist or his digits to her phone.

“I was disappointed that she doesn’t talk before or after her shows because she has to save her voice for the 44 songs she sings,” he said. “She doesn’t meet
anybody, or at least she didn’t want to meet me, so I took it personal[sic].” At this point, it seems like any attraction between the two was one-sided. Kelce’s
attempts to make proper contact with Swift at the show ultimately failed. Was she just not that into him? But Kelce, exhibiting the tenacity typical of a
professional sportsman, remained determined to get her attention. Wildest Beards The Kelce brothers couldn’t stop talking about Swift, even though she’d left
Travis hanging at the concert. Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?
</item 46>

<item 47>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 47>

<item 48>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 48>

<item 49>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 49>

<item 50>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.
</item 50>
=== END ITEM id=98c1e244d9b54763 ===

=== ITEM id=6e6b6ff7b94e71d4 ===
Gold sentences and facts:
Sentence 0: Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a return to fitness, and has been included on the squad.
  Fact 0.0: Johnny Cardoso had to pull out of the September camp due to an ankle injury.
  Fact 0.1: Johnny Cardoso has returned to action with Brazilian club Internacional.
  Fact 0.2: Johnny Cardoso's return to action indicates a return to fitness.
  Fact 0.3: Johnny Cardoso has been included on the squad.
Sentence 1: He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.
  Fact 1.0: Johnny Cardoso and his fresh legs are back just in time.
  Fact 1.1: Rookie sensation De'Von Achane hit Injured Reserve this week.
  Fact 1.2: De'Von Achane hit Injured Reserve with a knee injury.

Retrieved items:
<item 1>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Chase put a scare into the fantasy community on Wednesday when he was listed as a DNP, but after logging back-to-back limited practices, there is a lot more
optimism heading into Sunday. With Tee Higgins (hamstring) already ruled out, Cincinnati will be hoping Chase can feel healthy enough to play when they host the
Texans at 1 p.m. ET on Sunday.

The Bengals, despite getting a win over the Bills last Sunday, are a beaten-down bunch. Their defense will also be without a prominent edge rusher, Sam Hubbard,
complicating matters for a team hoping to stack wins as they compete for an AFC North crown. On the bright side, Joe Burrow appears to be finally healthy and
should have no problems targeting Tyler Boyd, Trenton Irwin, and Andrei Iosivas in the passing game. If Chase does end up suiting up, fantasy owners should
enjoy an even larger workload than the receiver already had.
</item 1>

<item 2>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
ET. We continue to stand by for more official updates from Canada Soccer.

If played, the rain-soaked conditions could rival those of the match Canada played in Honduras in the summer of 2022, which was wet and wild.

50 mins 1 hour 50 mins to potential kick: ** Tonight's match will not kick off on time**

Our first true update from Canada Soccer concerning tonight's match states that kickoff has been pushed back at least an hour. The statement claims that the
match will not kickoff before 8 p.m. ET, language which seems to leave the possibility open for further postponements or even calling off the game. We will keep
you updated as we get more information.

UPDATE: Tonight's #CanMNT vs. Jamaica @CNationsLeague Match will not kickoff before 8PM ET due to "on-field issues"
</item 2>

<item 3>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He was constantly unsure of where he should be as Germany built forward with incisive pace, and he was one of many U.S. players caught unaware on the first-half
Ilkay Gundogan goal.

McKennie didn't put a first-half pass wrong, going 18/18 before the break and 28/30 overall, but his positioning was all wrong out of possession. McKennie was a
sieve as the U.S. midfield was invisible defensively, leaving their defenders completely unshielded, particularly at the top of the penalty area.

It was only a matter of time before the floodgates opened, and that they did after halftime. McKennie was eventually withdrawn in what felt like one of Gregg
Berhalter's only tactical substitutions of the match.
</item 3>

<item 4>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
If the U.S. had profited more from his attacking presence the narrative around his performance might be different, but with the story of the match focused on
the defensive errors, Dest is a guilty party.

CM: Yunus Musah — 6

Another player who did well in possession but struggled defensively, Yunus Musah dazzled at times on the ball but was beaten on the first Germany goal by
whiffing with his slide.

Musah's nutmeg of Florian Wirtz was a highlight moment of the match, and he was active on both ends of the pitch. Yet a few bad giveaways or defensive moments
betrayed his usual stoutness in the middle of the pitch. Overall, the mistakes didn't take away from his performance quite as much as Dest's, but they were
still concerning.

CM: Weston McKennie — 4

Other than a pair of critical blocks right before halftime, Weston McKennie was a mess through the first 45 minutes.
</item 4>

<item 5>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
“I will commit to being the catalyst that will set the tone for our group for every game, to help restore our club to where it deserves to be, I am confident of
that. We will be back and better.”#NERevs pic.twitter.com/A0mVyGdaAZ — Seth (@SethMan31) November 10, 2023

Jamaica vs Canada lineups & team news

Jamaica manager Hallgrimsson also described his side as being on a mission, but they faced it without New York Red Bulls forward Cory Burke because of a groin
problem.

Karoy Anderson, who plays for English club Charlton Athletic, was called up for the first time. "I was just shocked," the midfielder told the Addicks' website.
"I was looking at it for a long time just taking it in and I told my mum and she started crying."
</item 5>

<item 6>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Gio Reyna looked sharp in the starting lineup for the United States men’s national team, in all the ways a young man can: his unyielding energy, his surprising
eagerness to cover the whole of the midfield, the periodic flashes of skill when the opportunity presented itself, even the bleach-blond hair that allowed
viewers to spot him in an instant.

There was one substantial element missing from his performance, though, in an immense friendly Saturday against Germany: being able to endure from starting to
finishing.

It was kind of like that for the USMNT as a whole, as they blew an early lead and fell, 3-1, in East Hartford, Conn.

MORE: USMNT undone by errors in loss to Germany

It is quite possible this was the most challenging friendly the U.S.
</item 6>

<item 7>
NFL Fantasy Football Week 5 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-04T21:56:25+00:00)
Achane is clearly far ahead of Mostert.

LAST WEEK: Before he fumbled the ball away twice, Mostert was splitting reps with De'Von Achane in a way that was unfavorable for him. Before the Dolphins got
inside the 10-yard line Mostert had a 16-to-9 edge, then Achane played every single snap inside the 10. Then after Mostert's second fumble of the half, Mostert
played just nine of the remaining 35 snaps including nothing near the end zone. Coach Mike McDaniel preached the importance of his offense not having
"self-inflicted wounds" immediately after the game.

LAST TWO WEEKS: Achane played all but two snaps inside the 10-yard line for Miami, so that seems like a big part of his role that will stick. As for third
downs, Mostert has a one-snap edge over Achane at 6 to 5, which means they're splitting those valuable touches.
</item 7>

<item 8>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Celtic defender had 22 touches and gave away possession just two times, completing two dribbles and making two clearances. Carter-Vickers was not a target
for Germany to attack nearly as much as the other three defenders on the pitch, a strong nod to his abilities.

65th min: Brenden Aaronson — 7

Probably the the U.S. substitute to make the biggest impact, Brenden Aaronson was bright in the attack, dazzling with his dribbling skills and helping usher the
U.S. forward amidst a second-half onslaught from Germany. A good showing from the Union Berlin winger who needs confidence at the moment.

65th min: Ricardo Pepi — 5

It wasn't an ideal situation for Ricardo Pepi to enter in, coming on with the U.S. shell-shocked from the quickfire Germany double. Still, he provided very
little up front, with just eight touches in 24 minutes of action.
</item 8>

<item 9>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: Breaking down Germany squad, tactics as Julian Nagelsmann debuts as manager

USA roster for October friendlies vs Germany, Ghana

The USMNT squad for the October friendlies was announced on Thursday, October 5. Gregg Berhalter has called in 23 players, which is one fewer than were brought
in for the friendlies against Uzbekistan and Oman last month.

Below is the official 23-man roster for the upcoming October friendlies against Germany and Ghana.

PSV midfielder Malik Tillman was forced to withdraw from the squad due to injury, with Alex Zendejas of Club America named to the roster in his place.

MORE: Tracking every goal scored by Christian Pulisic at new club AC Milan

Pos Name Club Caps Goals GK Ethan Horvath Nott. Forest (ENG) 9 0 GK Matt Turner Nott. Forest (ENG) 33 0 DEF Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino
Dest PSV Eindhoven
</item 9>

<item 10>
What USMNT can expect from Julian Nagelsmann, Germany in first match for new manager (Sporting News, 2023-10-06T20:56:33+00:00)
German national team squad for USA, Mexico friendlies

Pos Player Club Caps Goals GK Oliver Baumann Hoffenheim 0 0 GK Bernd Leno Fulham (ENG) 9 0 GK Marc-Andre ter Stegen Barcelona (SPA) 36 0 GK Kevin Trapp
Eintracht Frankfurt 7 0 DEF Robin Gosens Union Berlin 18 2 DEF Mats Hummels Borussia Dortmund 76 5 DEF David Raum RB Leipzig 18 0 DEF Antonio Rudiger Real
Madrid 62 2 DEF Niklas Sule Borussia Dortmund 47 1 DEF Jonathan Tah Bayer Leverkusen 17 0 DEF Malick Thiaw AC Milan (ITA) 2 0 MID Robert Andrich Bayer
Leverkusen 0 0 MID Julian Brandt Borussia Dortmund 44 3 MID Chris Fuhrich VfB Stuttgart 0 0 MID Leon Goretzka Bayern Munich 53 14 MID Pascal Gross Brighton
(ENG) 2 0 MID Ilkay Gundogan Barcelona
</item 10>

<item 11>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Gregg Berhalter and the U.S. men's national team continue their preparations for the 2024 Copa America and 2026 FIFA World Cup with friendlies against Germany
and Ghana this month.

First, the U.S. will take on Germany in Hartford, Connecticut on October 14. The 15th-ranked Germans are under new management, so the USMNT will serve as the
debut match for new head coach Julian Nagelsmann, the 36-year-old who has been out of a job since departing Bayern Munich back in March.

Most European countries are busy with Euro 2024 qualifying, but the Germans are hosting the tournament and therefore qualify automatically, opening up their
schedule for this friendly match.

Then, on October 17, the U.S. will travel to GEODIS Park in Nashville, Tennessee for a friendly against Ghana. The two nations have a deep history of
contentious matchups, including in three consecutive World Cup tournaments in 2006, 2010, and 2014.
</item 11>

<item 12>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🔚 pic.twitter.com/jSis5C72tj — Concacaf Nations League (@CNationsLeague) November 18, 2023

How can Canada qualify for the Copa America 2024?

The Rouges' task is now ostensibly straightforward: avoid defeat against Jamaica and they will be competing alongside the likes of reigning champions Argentina
at the Copa America in the US when it starts in June.

Jamaica have never won in Canada, although they showed enough here — in sunshine, mercifully, following the postponement of the game in torrential conditions
the previous night — to suggest they could spring an upset.

Their task will be made harder by the probable absence of Michail Antonio, while Bailey could not match speed and timing with accurate finishes when presented
with good early chances.

West Ham United forward Antonio immediately signalled for treatment before limping down the tunnel with a possible hamstring injury during the first half.

Reaction from Steven Vitoria:
</item 12>

<item 13>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Cameron Carter-Vickers Celtic (SCO) 12 0 DEF Sergino Dest PSV Eindhoven (NED) 28 2 DEF DeJuan Jones New England Revolution (USA) 7 0 DEF Kristoffer Lund Palermo
(ITA) 2 0 DEF Kevin Paredes Wolfsburg (GER) 1 0 DEF Tim Ream Fulham (ENG) 52 1 DEF Chris Richards Crystal Palace (ENG) 12 1 DEF Miles Robinson Atlanta United
(USA) 27 3 DEF Joe Scally Gladbach (GER) 6 0 MID Johnny Cardoso Internacional (BRA) 7 0 MID Luca de la Torre Celta Vigo (SPA) 17 0 MID Lennard Maloney
Heidenheim (GER) 0 0 MID Weston McKennie Juventus (ITA) 46 11 MID Yunus Musah AC Milan (ITA) 29 0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman
</item 13>

<item 14>
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
</item 14>

<item 15>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Even if one can argue he has the skill set to succeed at the No. 6, he did not appear comfortable transitioning on short notice.

MORE: USA vs. Germany player ratings

“Tyler’s a great player. It’s not nice to talk about guys that aren't here, though,” Berhalter told reporters afterward. “I’d rather talk about the guys that
are on the field.”

He didn’t, though, relative to how this group filled Adams’ huge vacancy. He was pleased to talk about the pressure the USMNT applied in the first half,
including a ball played over the top of Germany’s defense to Pulisic early in the game that was stopped by an offside call both player and coach disputed, and
Pulisic’s brilliant goal in the 27th minute that involved him driving to his right and blasting a curving shot while four defenders converged and ace goalkeeper
Marc-Andre Ter Stegen watched it sail into the upper righthand corner.
</item 15>

<item 16>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
So many national team fans sided with Reyna during the controversy that developed as the team competed at last winter's World Cup. They got what they argued for
in this game, but not what they wished.
</item 16>

<item 17>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The third German goal was emblematic of this lack of chemistry between the two U.S. center-backs: Richards stepped up to guard a free Niklas Fullkrug who got by
Sergino Dest, and Ream was unaware as that left Leroy Sane free on goal himself. These two just didn't have it, and Ream was powerless to pick up the slack.

Jamal Musiala dances through the USA defense and Germany lead 3-1 😮‍💨



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/PdjsjkObZR — B/R Football (@brfootball) October 14, 2023

LB: Sergino Dest — 4

If it were possible to give Sergino Dest two different scores, one for the attacking half and one for the defensive half, we'd do it, but that's not how this
works.
</item 17>

<item 18>
Fantasy Football Week 16 Wide Receiver Preview: Waiver adds, DFS plays, and more (CBSSports.com, 2023-12-19T20:51:32+00:00)
The Texans offense looks like one to avoid as long as C.J. Stroud is out. Ja'Marr Chase WR CIN Cincinnati • #1

Age: 23 • Experience: 3 yrs. Tee Higgins is a No. 2 receiver and Tyler Boyd is a fine flex.

WR Preview Numbers to Know

69.9% -- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets.

-- Nearly 70% of Rashee Rice's receiving yards have come after the catch, that's tops amongst wide receivers with at least 50 targets. 11.4 -- Deebo Samuel is
on pace to become just the fourth player since 1992 to average more than 11 yards per target with at least 60 targets in multiple seasons. The other three are
Tyreek Hill, Tyler Lockett, and DeSean Jackson.
</item 18>

<item 19>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
“I think he played a good game, showed his quality. I think everyone in the first half played well. So it was good to see him, good to have him on the field.”

It was not good to not have him on the field, because the collapse from a 1-1 halftime tie not only followed Reyna’s departure, but probably was connected. The
U.S. was able to threaten the opposing goal enough with Reyna on the field to cause the opposition to concern itself with defending more often than it
preferred. Replacement Luca de la Torre dulled the attack and did nothing to dissuade Germany’s rampage toward the American net.

There was plenty of blame to be shared on the goals by Niclas Fullkrug in the 58th minute and Jamal Musiala three minutes later.
</item 19>

<item 20>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Dest was lively up front as he combined with Christian Pulisic and worked very well on the ball. He got Mats Hummels booked on a counter-attack and delivered a
few very quality crosses.

Dest was also guilty of some of the game's worst defensive errors. He was completely non-existent as Leroy Sane blazed by him on the first Germany goal, and he
nearly got his team in trouble again right before halftime by holding onto the ball too long after making an interception rather than just clearing it out of
danger. Filling in for Antonee Robinson at left-back rather than his natural right, Germany targeted him on multiple occasions, to great profit.

He had a few good moments guarding Leroy Sane in the second half, but he was a top culprit for Germany's second goal by playing Niklas Fullkrug onside. If the
U.S.
</item 20>

<item 21>
Fantasy Football RB PPR Rankings Week 14: Who to start, best sleepers at running back (Sporting News, 2023-12-07T22:09:00+00:00)
(@ Browns) even in a tough matchup. Trevor Lawrence and his high-flying offense should dominate the time of possession against the Joe Flacco-led Browns,
leaving ETN plenty of opportunities to add to his per-game PPR average of 17.4 points (third best in the NFL entering Week 13).

Guess who's back? Back again? Achane's back — tell a friend! Stud rookie De'Von Achane (vs. Titans) went right back to roughing up the opposition in his Week 13
return from a knee injury that sidelined him one week prior. The potential league-winner casually put up 103 yards and two TDs on the Commanders, and we can't
help but place him and running mate Raheem Mostert in the top five this week despite a somewhat-tough matchup with Tennessee.
</item 21>

<item 22>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
Also missing is Antonee Robinson, who Gregg Berhalter said is dealing with a sports hernia, and while he played through it last camp and recently for his club,
it's affecting him and therefore rest is the best option here.

Johnny Cardoso had to pull out of the September camp due to an ankle injury, but he has returned to action with Brazilian club Internacional, indicating a
return to fitness, and has been included on the squad.

With Adams injured, there has been some noise on social media calling for Heidenheim midfielder Lennard Maloney to be brought into the squad in October, and he
was given the call-up here. The 23-year-old was born in Berlin to an American father, appearing for the U.S. at the U-20 level in the past and stating his
desire to reach the USMNT senior squad.

MORE: All players in U.S.
</item 22>

<item 23>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
🎥: @onesoccer pic.twitter.com/XzBldBSXfX — Canada Soccer (@CanadaSoccerEN) November 17, 2023

2 hours to kickoff: While the Canadian federation is reeling from the departure of John Herdman, one player is happy to see change happen at the top. Central
midfielder Mark-Anthony Kaye had been left out of multiple squads towards the end of Herdman's tenure, and he believes his relationship with the former Canada
head coach was the reason.

Me and John had some really tough conversations where he blatantly told me I wasn’t good enough to be part of the team. That was hard for me to accept,” Kaye
said to Sportsnet, after also posting to social media an honest self-reflection following his New England Revolution's elimination from the MLS playoffs a week
ago.

Heartfelt post by Mark-Anthony Kaye.
</item 23>

<item 24>
Top 25 MLB offseason trade candidates (Yardbarker, 2023-11-13T23:16:32+00:00)
17. Dylan Carlson, CF, Cardinals

It was a bit of a surprise that Carlson didn’t change hands over the summer. Various reports suggested the Cards were making him available as they looked for
ways to add a controllable starter. It didn’t happen, but the same logic applies this winter.

Carlson has been surpassed by Lars Nootbaar as the starting center fielder. St. Louis needs multiple starting pitchers and will likely bring in at least one via
trade. Carlson probably has less value than he did before the deadline. Not only did he finish with a mediocre .219/.318/.333 batting line, he underwent
season-ending surgery on his left ankle in September.

There’d nevertheless be teams willing to take a shot on a former top prospect who looked like an average or better center fielder 12 months ago. Carlson is
projected for a modest $1.8M salary and has three years of arbitration control.

18.
</item 24>

<item 25>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Especially when Canada were cruising with a 3-1 aggregate lead until the 62nd minute.

78 mins: GOAL - Canada 2-3 Jamaica (Bobby De Cordova-Reid), 4-4 aggregate

Jamaica are going through as it stands on away goals. The Fulham attacker converts from the spot after a handball from Eustaquio and they are back in front on
the night.

74 mins: Canada 2-2 Jamaica, 4-3 aggregate

Triple change for Canada. Junior Hoilett, Jonathan Osorio and Mark Kaye come on for Cyle Larin, Ismael Larin and Tajon Buchanan.

69 mins: GOAL - Canada 2-2 (Ismael Kone), 4-3 aggregate

Canada strike back. What an absolute rollercoaster. Davies floats in a deft cross and Kone's redirects it over Blake to send the crowd wild.
</item 25>

<item 26>
Fantasy Football RB Rankings Week 12: Who to start, best sleepers at running back (Sporting News, 2023-11-21T21:33:00+00:00)
However, even a moderate MCL sprain could mean a multi-week absence for the injury-prone veteran back, especially in a lost season for the 4-6 Pack. Rookie
Emanuel Wilson also hurt his shoulder in Week 11, so AJ Dillon could be getting the keys to Green Bay's backfield castle (although it's more of an oversized
shack, at this point).

Dolphins rookie De'Von Achane, who looked like the runaway Offensive Rookie of the Year earlier this season, looked to have aggravated his MCL injury upon his
Week 11 return against the Raiders. Achane has a chance to return for Miami's Week 12 tilt with the Jets on Friday, but we wouldn't be surprised if the 7-3
'Fins play it safe and keep him out until he's back to 100 percent. Meanwhile, Miami placed backup Salvon Ahmed on IR with a foot injury, so Jeff Wilson Jr.
could be back in play as a handcuff stash.
</item 26>

<item 27>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
The decision about how to replace Adams in the formation was complex — too much so to be resolved in one afternoon.

Reyna was exceptional throughout the first half: less influential than he might have been if given greater access to the ball but evincing a welcome taste for
covering ground to become helpful on defense or to retrieve the ball and launch an attack. Weston McKennie was his typical self, particularly when he could
connect with star winger Christian Pulisic along the left side.

Young Yunus Musah was deployed in the defensive midfield role, and it did not work. Whenever Germany were able to quell an American surge in the first half and
regain possession, they plowed with ease through through the center of the park. He is elite at maintaining possession when operating as a box-to-box
midfielder, and lately his club play, with Pulisic at AC Milan, has seen Musah in more of an attacking role.
</item 27>

<item 28>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
He was initially injured while playing for Leeds United back in March, and while he had finally recovered to make his Bournemouth debut in the EFL Cup in late
September, he played just 20 minutes off the bench before suffering a setback, sidelined again for the foreseeable future.

Gio Reyna has returned from his leg fracture and is called in for the first time under Berhalter since their falling out at the 2022 World Cup. Also included is
defender Cameron Carter-Vickers who only just made his debut for Celtic this season thanks to a hamstring injury. On the heels of that return just before the
international break, he is called in.

Goalkeepers Zack Steffen and Sean Johnson remain out of action due to injuries. Additionally, Josh Sargent is sidelined with a long-term ankle injury, while
Daryl Dike is also out after tearing his Achilles.
</item 28>

<item 29>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CB: Chris Richards — 4

Statistically, Chris Richards had a pretty good match, missing with just one pass of his 42 attempted and not being beaten at any point by any German attackers.
Yet his lack of volume speaks... well, volumes.

Richards was off the pace for much of his 65 minutes, and failed to get involved as his teammates were beaten repeatedly on the dribble. It was Richards'
positioning that let him down, and he should have been more involved with the U.S. under siege in such an open match.

CB: Tim Ream — 5

Chris Richards often left Tim Ream on an island, and the 36-year-old just could not cope with the waves of German attackers.

The third German goal was emblematic of this lack of chemistry between the two U.S.
</item 29>

<item 30>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
That would allow them to rotate Kamal Miller out, who picked up a yellow card in the first leg, keeping him safe for the semifinals.

In would step Alistair Johnston and Derek Cornelius, with Steven Vitoria retained alongside them.

The only injury absence for Canada at the moment is striker Lucas Cavallini, who withdrew from the squad prior to the two matches, replaced by Jacen
Russell-Rowe.

Canada projected starting lineup (3-5-2): Borjan (GK) — Cornelius, Vitoria, Johnston — Laryea, Eustaquio, Kone, Kaye, Davies — David, Larin.

Jamaica saw star striker Michail Antonio go down with a gutting ACL injury in the first leg, and he now begins a lengthy rehab process. West Ham are reportedly
furious about how it was handled, angered by both the conditions of the pitch in Jamaica and the fact he played another 10 minutes after sustaining the injury.
</item 30>

<item 31>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
ET, according to ESPN's Jeremy Fowler. The receiver was limited in practice thanks to the toe injury that has plagued him since leaving Week 8's game against
the Eagles, but barring a setback, he will return to action this weekend.

The receiver was working his way into flex territory with a couple of good performances before picking up the injury. However, we wouldn't consider him to be
worthy of a fantasy start this week unless you're in a bind. It's unclear whether Samuel will regain the same workload or if he'll be playing in limited action
as he works his way back. Jamison Crowder emerged in Samuel's absence and Jahan Dotson has seen his production drastically increase, as well. The wait-and-see
approach is the preferred one heading into Week 10.
</item 31>

<item 32>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
The Sporting News reviews how each USMNT player performed across the 90 minutes.

MORE: Recap the 3-1 defeat for the USMNT against Germany in Hartford

USA player ratings vs Germany

USA starting lineup (4-3-3, right to left): 1. Turner (GK) — 19. Scally, 4. Richards (Carter-Vickers, 66'), 13. Ream, 2. Dest — 7. Reyna (de la Torre, 46'), 6.
Musah, 8. McKennie (Cardoso, 75') — 21. Weah (Aaronson, 66'), 20. Balogun (Pepi, 66'), 10. Pulisic (Paredes, 75').
</item 32>

<item 33>
Fantasy Injury Updates Week 10: News on Justin Jefferson, Ja'Marr Chase, T.J. Hockenson & Curtis Samuel (Sporting News, 2023-11-11T21:17:00+00:00)
Jefferson (hamstring) is officially listed as "questionable" for Sunday's game against the Saints. Kickoff is slated for 1 p.m. ET, but fantasy owners won't
have to wait long for news on the star receiver. In order to be eligible to play, Jefferson will need to be activated by 4 p.m. ET on Saturday. Earlier in the
week head coach Kevin O'Connell said that it would be "a little aggressive" for Jefferson to get back on the field this week. However, the receiver's three
straight limited practices were enough for the team to at least weigh the decision, but thankfully, an early answer awaits.

WEEK 10 FANTASY ADVICE:

Sleepers | Busts | Projections | Start 'em, sit 'em

Jefferson's fantasy outlook is always through the roof, but it will be interesting to see if that changes with Josh Dobbs under center instead of Kirk Cousins.
</item 33>

<item 34>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
.@pulisic with a SCREAMER to open the scoring 😱 pic.twitter.com/KlG0oMeNPm — U.S. Men's National Soccer Team (@USMNT) October 14, 2023

“I think we created some problems,” Berhalter said. “And then just the threat we had a lot in the first half was really good. It put Germany on their heels at
times. But to me, it’s also about controlling the tempo of the game, and that’s what we really need to learn.”

It was a victory of sorts for Reyna to play 45 minutes after he’d gone only 27 in his one club appearance this season for Borussia Dortmund. He’d been injured
in June while playing for the U.S. and missed all of preseason training and nearly two months of the Bundesliga season.

“The goal is to send him back to Dortmund in a really good spot,” Berhalter said.
</item 34>

<item 35>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
24 mins: Jamaica 0-0 Canada

Substitute Trivante Stewart turns and tries his luck from distance for Jamaica, to no avail.

These sides are yet to find their range yet.

21 mins: Jamaica 0-0 Canada

Ismael Kone sends a fierce shot from outside the penalty area wide for Canada.

Trivante Stewart has replaced Jamaica's Michail Antonio, who limped off a few minutes ago.

18 mins: Jamaica 0-0 Canada

Bad news for West Ham United fans: Michail Antonio is down for an extended period of time and asks to be substituted.

He eventually rises to his feet and hobbles down the tunnel, accompanied by medical staff. Speculatively, that could be a hamstring issue.

15 mins: Jamaica 0-0 Canada

Leon Bailey should have scored for Jamaica again! An aerial ball down the flank bounces kindly for the Aston Villa attacker inside the penalty area on the left,
deceiving Steven Vitoria.
</item 35>

<item 36>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
He and his fresh legs are back just in time as rookie sensation De'Von Achane hit Injured Reserve this week with a knee injury.

DOLPHINS: Aren't quite the bastion of rushing that some headlines have made them out to be, but they're just over league average in rush rate at 43.7% and above
league average in RB runs per game at 22.6. All of these numbers are weighted heavily by their 50-point win over Denver in Week 3 when they ran the ball 42
times with their running backs. Their 7.6 yard per carry average is due to Achane's breakout start more than anything else.
</item 36>

<item 37>
Inter Miami vs FC Cincinnati live score, result, updates, highlights with Lionel Messi's MLS playoffs on line (Sporting News, 2023-10-07T21:30:00+00:00)
FC Cincinnati are otherwise light on availability concerns, with only a leg injury for Arquimides Ordonez and a long-term injury to young midfielder Stiven
Jimenez to worry about.

Talisman forward Luciano Acosta was partially rested midweek, coming off the bench to score the team's only goal against the Red Bulls in 32 minutes of play, so
he is ready to go as he chases the league's Golden Boot award. Also rested midweek was Alvaro Barreal who leads the team in minutes and also logged just 32 of
them on Wednesday.

FC Cincinnati starting lineup (5-3-2): Celentano (GK) — Arias, Hagglund, Miazga, Murphy, Barreal — Nwobodo, Acosta, Moreno — Vazquez, Santos.

FC Cincinnati subs (9): Kann (GK), Kubo, Mosquera, Angulo, Badji, Pinto, Halsey, Valenzuela, Boupendza.
</item 37>

<item 38>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Christian Pulisic with an absolute banger 🚀



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/G7t6KxLjFO — B/R Football (@brfootball) October 14, 2023

Player ratings for USMNT substitutes vs Germany

45th min: Luca de la Torre — 5

Brought on to replace Gio Reyna at halftime, Luca de la Torre was unable to stem the German tide, despite being a more defensive-minded option than Reyna in
what should have been a swap to shore up the midfield.

65th min: Cameron Carter-Vickers — 7

With Tim Ream and even Chris Richards struggling at various stages in this match, Cameron Carter-Vickers remains a top-tier center-back in this U.S. player pool
and a clear starter when healthy. He came off the bench due to his recent hamstring troubles, but was stout in his 25 minutes of action.
</item 38>

<item 39>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
are likely to play between now and the 2026 World Cup here in North America, and so it was viewed by coach Gregg Berhalter as a significant opportunity to learn
about his team. And what was learned, above all, was there needs to be an alternative approach to the midfield for whenever Tyler Adams is unavailable to play.

Adams is the one proven defensive midfielder — a No. 6, as they call it in soccer jargon — on the USMNT roster. He has been absent from the national team since
injuring his hamstring in the spring, and it’s possible now he’ll need a second surgery to address the issue.

The decision about how to replace him in the lineup was rather simple. It always was the plan for Reyna to play roughly half this game, and the obvious choice
in that circumstance, given the sometimes ridiculous controversy that developed from Reyna’s reserve role at the World Cup, was to have him start.
</item 39>

<item 40>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
CM: Gio Reyna — 7

In his first 45 minutes of international football under Gregg Berhalter since the World Cup fallout, Gio Reyna was his usual self, drawing the attention of
German defenders and proving visionary with his passing. He combined with Folarin Balogun on a number of occasions on the counter, which is a link-up that U.S.
fans have been begging for more of.

Reyna was only fit for 45 minutes of play here as he returns from a leg injury, but it's no coincidence that Germany truly secured midfield dominance when he
came off the field. A promising performance for the young playmaker who should head back to Borussia Dortmund after this international break with plenty of
confidence moving forward.

RW: Tim Weah — 8

Though Germany carved the U.S. defense to pieces, their attacking threat remained strong, and almost all the danger came from the wings.
</item 40>

<item 41>
Gio Reyna looks good in start for USMNT vs. Germany, but those involved in the finish are punished in 3-1 defeat (Sporting News, 2023-10-14T23:41:07+00:00)
Sergino Dest allowed the ball to deflect off his foot and into dangerous territory at the top of the box, de la Torre became a spectator with a too-great view
and central defender Chris Richards missed two opportunities to shut down the threat. McKennie, central defender Tim Ream and right back Joe Scally all bore
some responsibility for the second.

“When you watch the goals back — I had the opportunity to do that in the coaches’ office — it’s really these fine moments, these split seconds where you need to
be well-positioned, not open up any gaps in your back line,” Berhalter said. “And that’s where it went wrong.”

With the attempt to transform Musah into some semblance of Adams, though, it almost was certain to go wrong. The USMNT conjured a more compelling attack because
of Reyna’s presence, but the ability to control the game went missing.
</item 41>

<item 42>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
While Christian Pulisic starred on the left, Tim Weah was equally devastating down the right, dazzling with electric pace and good link-up.

Weah's biggest moment came when he scorched Robin Gosens on the wing in the first half, knocking it past the German defender and then beating him with
blistering speed to get back on the ball. Weah's statistical haul don't quite do him justice, but he was a threat throughout his 66 minutes of action.

ST: Folarin Balogun — 7

U.S. fans are salivating for Folarin Balogun to get more touches up front, and as Gregg Berhalter foreshadowed pre-match, they made a clear attempt to do that.
Gio Reyna and Tim Weah did the most of the work in that regard,

LW: Christian Pulisic — 9

There's no doubting who remains the most electric and incisive U.S.
</item 42>

<item 43>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Antonio was replaced by Salernitana striker Trivante Stewart, but he struggled and was himself withdrawn in the 81st minute. Demarai Gray could step in, but he
himself is struggling for fitness, and while he remains with the team he was not part of the first leg matchday squad. Another option is Colorado Switchbacks
forward Romario Williams, who scored 15 goals in 29 USL Championship matches this season.

Other injuries for Jamaica to contend with include Cory Burke and Amari'i Bell, who both miss out on this international window.

Jamaica projected starting lineup (4-3-3): Blake (GK) — Leigh, Bernard, Pinnock, Lembikisa — Latibeaudiere, D. Johnson, Bailey — Reid, Nicholson, R. Williams.
</item 43>

<item 44>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
With Berhalter having since confirmed the two have broken the ice and talked over the phone, they will now look to move forward without any more personal
conflict.

After the October roster release, Berhalter confirmed that the two spoke "weeks ago" on Zoom, with the U.S. head coach appreciative of Reyna's willingness to
move forward. He spoke that because Reyna has not yet played for his club this season, their goal will be to "get him minutes" so he can build his fitness and
return to Dortmund in the best position possible to begin his club campaign.

Berhalter stated that they see Reyna being able to play three different positions: central midfielder, attacking midfielder, and winger, with further
clarification being his skill-set fitting a "between the lines" style of wide play if he were deployed on the wing, as opposed to one glued to the touchline.
</item 44>

<item 45>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
He featured his usual pressing, and came close to scoring once on a messy chance that Germany just barely managed to clear, but otherwise he didn't quite cause
the opposing back line problems to open spaces up for teammates.

75th min: Johnny Cardoso — 6

Johnny Cardoso only had 15 minutes at the end to showcase his abilities, but it begged the question why he or Lennard Maloney weren't the ones introduced at
halftime instead of Luca de la Torre. Cardoso lost possession just once in his 10 touches and completed a tackle, although he was also dribbled past a few times
as well.

75th min: Kevin Paredes — 5

With 10 touches in his 15 minutes of action towards the end of a mucked up game of numerous substitutions, Kevin Paredes didn't stand out, losing possession
four times and completing four passes. Tough to evaluate his time on the field, but he didn't stand out.
</item 45>

<item 46>
Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg (Sporting News, 2023-11-17T21:48:03+00:00)
Watch:

Jonathan David lights up the scoreboard for Canada! 🇨🇦 pic.twitter.com/bCVlAEQpge — Concacaf Nations League (@CNationsLeague) November 18, 2023

45 mins: Jamaica 0-0 Canada

End-to-end fare. Ethan Pinnock stops Jonathan David from getting a shot away inside the penalty area, and Andre Blake then punches Alphonso Davies' cross clear
from a crowded box.

At the other end, Sam Adekugbe shows strength and speed to stop Trivante Stewart from racing clear onto a long ball.

44 mins: Jamaica 0-0 Canada

Bobby De Cordova-Reid almost set up an opener for Leon Bailey five minutes ago. This time he takes aim himself from a long way out, zipping a shot into Milan
Borjan's arms.

41 mins: Jamaica 0-0 Canada

Milan Borjan is down for an extended period of treatment.
</item 46>

<item 47>
Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg (Sporting News, 2023-11-21T22:35:30+00:00)
Jamaica's first-ever victory on Canadian soil has seen the Reggae Boyz fight back to reach the CONCACAF Nations League semifinal and qualify for Copa America
2024.

A 3-2 win in the second leg at BMO Field in Toronto allowed Jamaica to progress on away goals after a 4-4 aggregate tie across the two legs.

After Canada opened the scoring through Alphonso Davies they held a comfortable 3-1 aggregate lead, but that advantage eroded in 15 second-half minutes as
Jamaican striker Shamar Nicholson scored a brace, before Fulham attacker Bobby De Cordova-Reid converted the penalty that sealed progression.

Canada will now have to defeat Trinidad and Tobago in a one-off playoff in March to reach the Copa America 2024 tournament to be hosted in the U.S.

MORE: Highlights, recap of Canada's narrow win in the first leg

The Sporting News followed the Canada vs. Jamaica match live, providing score updates, commentary and highlights as they happened.
</item 47>

<item 48>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
MORE: All players in U.S. national team pool currently playing for a European club

Gio Reyna to make USMNT return under Gregg Berhalter

The ability of Berhalter and Reyna to work together moving forward has been a major talking point since the fallout of their saga after the 2022 World Cup.

Reyna's lack of playing time at the World Cup was exacerbated by the actions of his parents in the weeks that followed, which sparked a domestic violence
investigation of Berhalter before his eventual re-hire as national team head coach.

Now, having recovered from a leg fracture suffered in summer Nations League duty under an interim USMNT head coach, the 20-year-old Borussia Dortmund playmaker
returns to the international fold under Berhalter for the first time.

The USMNT head coach has been forthcoming about the situation, admitting upon his re-hire that the two have a "lot of work to do" but insisting time was needed
to broach the situation.
</item 48>

<item 49>
USA vs Germany player ratings: USMNT defensive errors blight positive performance by Pulisic, Weah (Sporting News, 2023-10-14T22:46:31+00:00)
Matt Turner says no ❌



Watch USA vs. Germany live on TNT or Max 📺 pic.twitter.com/x2HslCk3s2 — B/R Football (@brfootball) October 14, 2023

RB: Joe Scally — 6

Joe Scally was tidy in possession, and while he offered very little in the attack, that was largely by design as they didn't need him on that right side with
Tim Weah cooking.

Scally defended ok, and wasn't guilty of any big mistakes, which is more than most of the other U.S. defenders can say. It wasn't a standout performance by any
means, and the whole back line has some soul searching to do, but others were far more culpable than the Borussia Monchengladbach man.
</item 49>

<item 50>
USA squad for October friendlies 2023: USMNT roster for friendly matches vs Germany, Ghana (Sporting News, 2023-10-04T22:42:00+00:00)
0 MID Gio Reyna Borussia Dortmund (GER) 20 4 MID Malik Tillman PSV Eindhoven (NED) 6 0 FWD Brenden Aaronson Union Berlin (GER) 34 8 FWD Folarin Balogun Arsenal
(ENG) 4 2 FWD Ricardo Pepi PSV Eindhoven (NED) 18 9 FWD Christian Pulisic AC Milan (ITA) 62 26 FWD Tim Weah Juventus (ITA) 33 5 FWD Alex Zendejas Club America
(MEX) 7 1

Tyler Adams, Antonee Robinson injured for October camp

Gregg Berhalter largely has the entire first-choice player pool available for selection, with only a few injuries to work around.

There is one notable absence, as captain Tyler Adams remains sidelined with a frustrating long-term hamstring injury.
</item 50>
=== END ITEM id=6e6b6ff7b94e71d4 ===
