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

=== ITEM id=d170cf22c7f3786b ===
Gold sentences and facts:
Sentence 0: He revealed that he made a friendship bracelet, a fan tradition at her concerts, with his number on it to give to her, seemingly expressing his interest in Swift.
  Fact 0.0: A man revealed that he made a friendship bracelet.
  Fact 0.1: Friendship bracelets are a fan tradition at Taylor Swift's concerts.
  Fact 0.2: The friendship bracelet had his number on it.
  Fact 0.3: He made the friendship bracelet to give to Swift.
  Fact 0.4: The friendship bracelet seemingly expressed his interest in Swift.
Sentence 1: On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24 September.
  Fact 1.0: The man later invited Swift to one of his games on his podcast.
  Fact 1.1: Swift is the Anti-Hero singer.
  Fact 1.2: The game was at Arrowhead Stadium.
  Fact 1.3: Swift accepted the invitation on 24 September.
  Fact 1.4: Swift famously accepted the invitation.
Sentence 2: Now, Swift has shared that she has nothing to hide in her new relationship with Kelce.
  Fact 2.0: Swift has shared that she has nothing to hide.
  Fact 2.1: Swift is in a new relationship with Kelce.

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
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 3>

<item 4>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“The Taylor Swift and Travis Kelce news has been a pop cultural moment we’ve leaned into in real time, as it’s an intersection of sport and entertainment, and
we’ve seen an incredible amount of positivity around the sport.”

“The vast majority of our content has remained focused on the game, our players and variety of other initiatives, including our Toy Story Funday Football
alt-cast, the international games and more,” they continued.
</item 4>

<item 5>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
87 with gusto.

The 12-time Grammy winner appeared to shout “let’s f—ing go” after Kelce scored a touchdown before chest-bumping other Chiefs fans. Considering Swift is known
for being an Eagles fan – she’s originally from Pennsylvania and even references the team in Gold Rush – her outward support for the Kelce’s team seemed extra
telling. Swifties were immediately on high alert, with some capturing Swift and Kelce leaving the stadium together. If that wasn’t enough, extra observant fans
were able to note that Kelce’s blue and white post-game outfit was supposedly labelled the “1989 bedroom painting set”, perhaps an intentional reference to
Swift’s 2014 album, which she will be re-releasing in October. Swift’s penchant for Easter eggs must be rubbing off on the tight end player. As if this weren’t
enough subtle references already, they were also spotted in a “getaway car” together outside the stadium – potentially a nod towards her 2017 track.
</item 5>

<item 6>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Unfortunately, the bracelet never made it to her wrist or his digits to her phone.

“I was disappointed that she doesn’t talk before or after her shows because she has to save her voice for the 44 songs she sings,” he said. “She doesn’t meet
anybody, or at least she didn’t want to meet me, so I took it personal[sic].” At this point, it seems like any attraction between the two was one-sided. Kelce’s
attempts to make proper contact with Swift at the show ultimately failed. Was she just not that into him? But Kelce, exhibiting the tenacity typical of a
professional sportsman, remained determined to get her attention. Wildest Beards The Kelce brothers couldn’t stop talking about Swift, even though she’d left
Travis hanging at the concert. Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?
</item 6>

<item 7>
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
</item 7>

<item 8>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Singer Taylor Swift has a blank space, and she's writing Travis Kelce's name. There were rumors the two were dating for weeks, and those talks only intensified
with swift attending five Chiefs games so far this season, including Sunday's game agains the Chargers.

The Era's Tour singer was seen chatting with the tight end's mom, Donna Kelce, and cheering emphatically when No. 87 scored a touchdown in the first game she
attended last month. Last week, she was in attendance to watch the Chiefs as they defeated the Denver Broncos in Week 6 on "Thursday Night Football."

She is once again at Arrowhead Stadium, this time to watch the Chiefs take on another AFC West opponent, the Los Angeles Chargers.

For those wondering how this apparent relationship came to be, wonder no longer. We have created an extensive timeline of the high-profile couple that is
currently the talk of the league, and even added some Swift lyrics to go along with some of the highlights.
</item 8>

<item 9>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“I placed my hand on the gentleman’s back to let him know I was behind him. If I would have pushed him, he probably would have turned around and tased me,” he
said.

The Independent has contacted representatives for Taylor Swift and Travis Kelce for contact.
</item 9>

<item 10>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 10>

<item 11>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 11>

<item 12>
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
</item 12>

<item 13>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
ADVERTISEMENT Article continues below this ad

How can you not love Taylor Swift? Most of the 12 Grammy Awardee’s songs are about that warm fuzzy feeling and chemical reaction that we call love. And her
songs have earned the support of 103.7992 million Swifties in the world. Yeah, we don’t have a round figure for that. We have the exact figure up to four
decimals. That’s how much data we have on her. Football has 184 million fans. The league is able to generate a revenue of $18 billion from 184 million. If it
adds 104 (after rounding 103.7992) million to that number, guess how much revenue it will generate? $28 billion. And just like that, their target will be met.
</item 13>

<item 14>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“OF ALL 230 SONGS IN THE DISCOGRAPHY I’M SICK,” one fan commented, while another person wrote: “It’s unbelievable how invested I am in this.”

“I’m gonna need to go back and see all photos of him from the last three months and dissect if he’s wearing friendship bracelets,” a third wrote.

In other photos from the charity event - which were shared to Instagram by Jordan’s KC photography - Kelce wasn’t seen wearing the friendship bracelet, leading
many people to believe that he received the bracelet from a fan that day.

Two days after his charity event, the “All Too Well” singer was spotted cheering for the Kansas City Chiefs tight end from the box seats at Arrowhead Stadium in
Kansas City, Missouri. Not only was she seen sitting next to Kelce’s mother, Donna, but she also appeared to yell “let’s f***ing go” in response to Kelce’s
third-quarter touchdown reception.
</item 14>

<item 15>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
The now-viral photo shows how Kelce may have hinted at the ongoing dating rumours before Swift attended his game. The snap showed Kelce posing at the Kelce Car
Jam charity event on Friday 22 September, where he was sporting a beaded friendship bracelet that read: “Vigilante S***”

The beads on the friendship bracelet appeared to reference Swift’s song “Vigilante S***” from her 2022 album, Midnights. In addition to the Swiftie-themed
jewellery, Kelce also wore a white striped shirt with silver chain necklaces, black sunglasses, and black pants for the event.

In the comments, fans both praised and questioned Kelce for the beaded friendship bracelet. Many people also pointed out how fans of the “Bad Blood” singer
famously trade and wear friendship bracelets at Swift’s Eras Tour concerts.
</item 15>

<item 16>
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
</item 16>

<item 17>
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
</item 17>

<item 18>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 18>

<item 19>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Here's a look at how it all started and where it has gone since:

June 29: Kelce makes friendship bracelets for fellow TEs

Is this where it all begin? Kelce made a friendship bracelet to give to the newcomers at a camp for NFL tight ends. He said he was "taking a page out of Taylor
Swift's book."

July 26: Kelce makes friendship bracelet for Swift

"So make the friendship bracelets, take the moment and taste it." - You're On Your Own Kid

Kelce announced that he went to Swift's highly anticipated concert, but said he did not have the chance to meet the 12-time Grammy winner.
</item 19>

<item 20>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 20>

<item 21>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
The dating speculation first began in July, when Kelce confessed that he tried and failed to give Swift a friendship bracelet with his number on it during her
Eras Tour show. On his podcast, he later invited the “Anti-Hero” singer to one of his games at Arrowhead Stadium - an invitation she famously accepted on 24
September. Since then, Swift has attended three more of Kelce’s games.

Kelce showed his support for Swift when he attended her Eras Tour show in Buenos Aires, Argentina, on 11 November. During the concert, Swift gave a shout-out to
the Chiefs player when she changed her “Karma” lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang,
letting out a laugh as fans in the audience screamed in response. The “All Too Well” singer was later seen running into his arms and kissing the NFL player
following the concert, as he whisked her away backstage.
</item 21>

<item 22>
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
</item 22>

<item 23>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“I threw it out there, I threw the ball in her court,” he said. “I told her, you know, I’ve seen you rock the stage in Arrowhead [Stadium], you might have to
come see me rock the stage in Arrowhead and see which one’s a little more lit. So, we’ll see what happens in the near future.”

Back in July, Kelce first confessed that he tried - and failed - to give Swift his number by giving her a friendship bracelet at one of her Eras Tour concerts.
“I was disappointed that she doesn’t talk before or after her shows, because she has to save her voice for the 44 songs that she sings,” he explained to his
brother on their podcast.

“So I was a little butthurt I didn’t get to hand her one of the bracelets I made for her,” Kelce said.
</item 23>

<item 24>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She explained that she doesn’t necessarily pay attention to the cameras and fans around her in those stadiums because she is watching her boyfriend play.

(Denver Post via Getty Images)

“I’m just there to support Travis,” she said. “I have no awareness of if I’m being shown too much and pissing off a few dads, Brads, and Chads.”

When discussing how many Swifties have now started watching NFL games since she started dating Kelce, she quipped about how much she likes the sport. “Football
is awesome, it turns out,” she joked. “I’ve been missing out my whole life.”

Swift’s effect on the NFL didn’t go unnoticed in October, as one game between the Chiefs and New York Jets - which she attended with some of her famous friends
- became the most-watched Sunday NFL show since the Super Bowl LVII in February 2023.
</item 24>

<item 25>
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
</item 25>

<item 26>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
After the game, which saw the Chiefs beat the Chicago Bears 41-10, Kelce was seen leaving the stadium in a blue and white splatter-paint denim suit. Eagle-eyed
fans were quick to believe that Kelce’s outfit paid homage to Swift’s forthcoming album, 1989 (Taylor’s Version), which is due for re-release on 27 October.

After the two escaped the crowd at Arrowhead Stadium in Kelce’s convertible, according to photos and videos posted online, the football star reportedly rented
out Prime Social Rooftop in Kansas City for a post-game celebration with his family, his teammates, and Swift.

“Travis bought out the restaurant for his family and team. Taylor arrived, wearing a denim dress, and was seen snacking, having some cocktails and dancing
alongside Travis,” the witness told Entertainment Tonight. “The two were very affectionate with one another but kept things fun and lighthearted.
</item 26>

<item 27>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 27>

<item 28>
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
</item 28>

<item 29>
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
</item 29>

<item 30>
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
</item 30>

<item 31>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Could they have been driving off into the sunset? Of course, none of this actually confirms Swift and Kelce are actually dating. But regardless of whether it’s
a friendship or a romantic relationship, it has sent shockwaves through two massive cultural franchises, expanding the NFL’s audience into uncharted territory.
It’s a cultural pairing on par with – and perhaps even bigger than – Kim Kardashian and Kanye West.
</item 31>

<item 32>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Travis’ teammates also showed up to the after-party, as well as his mom, Donna Kelce. The party lasted until 2 am.”

The late-night outing came days after the NFL star officially addressed ongoing dating rumours about him and the “Anti-Hero” singer. During an episode of The
Pat McAfee Show on 21 September, he mocked his brother Jason Kelce for joking that he “100 per cent” believed his younger sibling was dating Swift.

“It’s hilarious how much traction this has actually gotten, right now it’s like a game of telephone,” Kelce said. “Where everybody’s whispering in each other’s
ears and hearing random stuff. No one actually knows what’s going on, especially when you got Jason Kelce on live television telling people just both sides - he
doesn’t know, it’s true, this and that.”

He also noted that he was still open to pursuing a relationship with Swift, revealing that he subtly asked if she’d want to attend one of his football games.
</item 32>

<item 33>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
The “All Too Well” singer’s interview with Time marks her first time publicly speaking out about Kelce, who she’s been romantically linked to since this summer,
when the football tight end first confessed to attempting to give her his number.

While Swift has attended a few of Kelce’s games since their relationship started, he’s also gone on to support her career. In November, he made his way to
Buenos Aires, Argentina, to attend one of her Eras Tour shows. During the concert, Swift gave a shout-out to the Chiefs player when she changed her “Karma”
lyrics to reflect their relationship. “Karma is the guy on the Chiefs, coming straight home to me,” she sang, letting out a laugh as fans in the audience
screamed in response.

After the show, the “Style” singer was seen running into Kelce’s arms and kissing him, before he whisked her away backstage.
</item 33>

<item 34>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Oct. 15: Kelce, Swift make 'SNL' cameos

The two were spotted out in NYC together and then had a surprise appearance on "Saturday Night Live." Kelce had a cameo in a skit about how much coverage Swift
was getting during games, while the singer introduced the musical act of the night, Ice Spice.

Oct. 22: Swift attends fifth Chiefs game

Swift has now attended games broadcast across all five networks (CBS, NBC, Fox, ESPN and Amazon). This week, she is watching the Chiefs face the Chargers and
will be shown on CBS. She is sporting a Chiefs sweatshirt and is seated next to Kansas City quarterback Patrick Mahomes' wife, Brittany.
</item 34>

<item 35>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Not to mention her rumoured relationship with Kansas City Chiefs tight end Travis Kelce caused a 400 per cent spike in NFL jersey sales.

Since her first NFL appearance in September, Swift’s occasional attendance at Kelce’s games has received pushback from NFL fans who believe her presence is
distracting from the game itself. “I’m just there to support Travis,” Swift admitted. “I have no awareness of if I’m being shown too much and pissing off a few
dads, Brads, and Chads.”

She playfully added: “Football is awesome, it turns out. I’ve been missing out my whole life.”

The “Bad Blood” singer revealed that their romance began when Kelce gave her shoutout on his podcast, admitting he attempted to give Swift his phone number
after one of her Eras Tour shows. The pair immediately began “hanging out” and by the time she attended her first Chiefs game in September, they were already a
couple. “I think some people think that they saw our first date at that game?
</item 35>

<item 36>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I appreciate you always trying to make it to both of our games.”

“She also told me thank you because she was in an alternate universe, she was sitting there with Deadpooland Wolverine,” Kelce added, referencing Reynolds’ and
Jackman’s respective superhero movie characters. “It was a hot ticket, man, you know what I’m saying? That suite was rockin’.

“They were enjoying themselves. Somebody told me that everybody was enjoying themselves in that suite,” Kelce said.

Not only has NFL viewership increased since Swift began attending her rumoured boyfriend’s games, but jersey sales for the Chiefs tight end have reportedly
spiked by 400 per cent, according to a report from the NFL’s e-commerce partner, Fanatics.

Following critcism the league is focusing too much on Swift, the NFL responded with an official statement handed to People on 4 October.

“We frequently change our bios and profile imagery based on what’s happening in and around our games, as well as culturally,” the message read.
</item 36>

<item 37>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Well I know for DAMN SURE that s*** about ‘having kids within the next year’ is NOT coming from Taylor’s side,” one claimed. “She’s got the Eras Tour booked
all through next year, and she’d rather die than cancel a show. But we do know who’s been public about looking for a ‘BREEDER.’”

The comments about Travis and Swift’s romance came days after they seemingly confirmed their relationship. On 22 October, Swift made her return to Arrowhead
Stadium to watch as the Kansas City Chiefs won their game against the Los Angeles Chargers. The occasion marked Swift’s fourth time attending one of Kelce’s
games, amid ongoing reports that they are dating.

Following the event, Chariah Gordon, the partner of Kelce’s teammate Mecole Hardman Jr, took to Instagram to share a snap of the singer and NFL player’s sweet
PDA moment. In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.
</item 37>

<item 38>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 38>

<item 39>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 39>

<item 40>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
This was the first, albeit tentative, confirmation from the media rather than from amateur Swiftie sleuths on social media. So, it was only a matter of time
before it entered the cultural zeitgeist. Blank Space Which it did, very quickly. On September 17, Kelce scored a touchdown against the Jaguars, triggering CBS
announcer Ian Eagle to reference Swift’s 2014 hit, Blank Space.

“Kelce finds a blank space for the score,” Eagle said, poking fun at Kelce’s newfound Swiftie status. And Eagle wasn’t the only one, with NFL Network’s Rich
Eisen including a slew of lyric references during a preshow, including “shake it off”, “anti-hero” and “all too well”. Kelce even acknowledged the
tongue-in-cheek banter, commenting on a montage video of the preshow that Eisen posted to Instagram. “Well played Rich … Well played,” he wrote. Suddenly,
football fans and Swifties were living through this surprising collision.
</item 40>

<item 41>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Yeah, my personal life -- that's not so personal. I did this to myself Jason, I know this. What's real is that you know it is my personal life. ... She's not
in the media as much as I am doing this show every single week."

Sept. 27: Jerry Jones comments on the Swift effect

Swift is getting the owners talking, too.

"It is hard to move that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a
lot of eye balls. She blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core
constituency. We all, I think, felt the same way when she's sitting up there with [Travis Kelce's] mom and she's going nuts when he's playing and those hearts
are popping out everywhere.
</item 41>

<item 42>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Fans have spotted Travis Kelce wearing a friendship bracelet with Taylor Swift lyrics on it, just days before she attended his Kansas City Chiefs game.

In a video posted to TikTok on 25 September, a fan shared a photo of Kelce posing with the piece of jewellery on his wrist. The post came after Swift was seen
enthusiastically cheering him on in the box seats at Arrowhead Stadium, fuelling speculation that she and the athlete are dating.
</item 42>

<item 43>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 43>

<item 44>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
In the picture, Swift could be seen standing next to Kelce, reaching to kiss his cheek.

In addition to the kiss, the football star also had the perfect reaction to a fan calling him Swift’s boyfriend. After a fan shouted: “Let’s go Taylor’s
boyfriend” while Kelce was waiting for an elevator, he responded with a small smile on his face and by pumping his fist into the air.

Aside from their appearances at sports events, Swift and Kelce have also been photographed together on other occasions, including a date earlier this month in
New York City after their surprise appearances on the season 49 premiere of Saturday Night Live. At the time, sources alleged to Entertainment Tonight that,
during the date, Kelce got a little too close to Swift’s security personnel and allegedly told her bodyguard to step aside.

However, during an episode of his podcast, Kelce explained what he said to the security, and clarified that he didn’t push the bodyguard.
</item 44>

<item 45>
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
</item 45>

<item 46>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.
</item 46>

<item 47>
Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game (The Independent - Life and Style,
2023-09-26T22:05:34+00:00)
“If you’re up on Taylor Swift concerts, there are friendship bracelets and I received a bunch of them being there, but I wanted to give Taylor Swift one with my
number on it.”
</item 47>

<item 48>
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
</item 48>

<item 49>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
The first link between Swift and Kelce was established in 2016, when Kelce – who helped bring his team to Super Bowl victory in both 2020 and 2023 – played the
classic game “Kiss, Marry, Kill”. Speaking with AfterBuzz TV, Kelce was given the options of Swift, Katy Perry and Ariana Grande. He said he’d ultimately kill
Grande, marry Perry and kiss Swift. That was about six years ago, but it has more recently set tongues wagging as fans wondered how long the plan for them to
meet has been in the works. Cruel Summer Fast-forward seven years, and things started to heat up. In July, speaking on the podcast he shares with his brother,
New Heights, Kelce said he intended to give Swift a friendship bracelet with his number on it during her Eras Tour concert in Kansas City. Unfortunately, the
bracelet never made it to her wrist or his digits to her phone.
</item 49>

<item 50>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Suddenly, football fans and Swifties were living through this surprising collision. Swift lyrics were infiltrating NFL game commentary, and her fans were
furiously becoming football experts. All that needed to happen was for Swift herself to infiltrate a game, determining once and for all whether there was
anything to these rumours or not. This Is Me Trying

On September 21, Kelce announced on The Pat McAfee Show he had invited Swift to watch one of his games. “I told her, ‘I’ve seen you rock the stage in Arrowhead,
you might have to come see me rock a stage at Arrowhead and we can see which one’s a little more lit,’” he said. “We’ll see what happens in the near future.”
Were they finally about to hard-launch their friendship/potential romance to the world, laying all speculation to rest? Love Story This finally brings us to the
fated sighting of Swift at Arrowhead Stadium on Monday, cheering on No. 87 with gusto.
</item 50>
=== END ITEM id=d170cf22c7f3786b ===

=== ITEM id=7c233b9758b3eb02 ===
Gold sentences and facts:
Sentence 0: “It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game in junior — it indicated he was going to be able to do that at some point in the NHL.
  Fact 0.0: Davidson said the player was not playing in the NHL in the last few years.
  Fact 0.1: Davidson said the level of performance showed the player could control a game and dominate a game in junior.
  Fact 0.2: Davidson said this performance indicated the player was going to be able to do that at some point in the NHL.
Sentence 1: There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala, among many others.
  Fact 1.0: The player helped guide more than a few future NHL stalwarts at that place.
  Fact 1.1: Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor Arvidsson, and Kevin Fiala were among the future NHL stalwarts guided there.
  Fact 1.2: Many other future NHL stalwarts besides those named were also guided there.

Retrieved items:
<item 1>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“To me, when he takes a shot, it looks like he just bends down, picks it up with his hand and places it where he wants to.

“He believes that ‘Hey, f— it, the puck is only two and a half inches wide, I can put it wherever I want.'”

Today, Turk would take Bedard’s shot against anyone’s.

Advertisement

“On a shot release basis only, if I had to bet on who could get the puck off the quickest, with the most deception, with optimal speed, power and accuracy based
on a starting point to a finishing point, I’m picking Connor Bedard over Auston Matthews,” Turk said. “And I’m not taking anything away from Auston. It’s just a
little bit different because one’s a righty, one’s a lefty and one’s got a little bit of a higher-angle pull-in change.”

If there’s one thing Nick Quinn wants people to know about “The Bedard,” it’s that it isn’t just some natural gift.
</item 1>

<item 2>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“It’s a lot of different angles he can give you.”

Sabres first-round pick Zach Benson, who has played on teams with and against Bedard and skates with him in the summers, says it’s Bedard’s ability to set up
the shot with his agility pre-shot that stands out to him.

“He can move left to right like no one I’ve seen before,” Benson said.

“It’s just trying to get out, face him, and hope it hits ya,” said Scott Ratzlaff, Seattle Thunderbirds goalie and Buffalo Sabres prospect, with a smile. “He
can shoot from anywhere and he’s lethal from anywhere. You’ve just always got to be ready just in case he shoots it. And then he’s got a really good toe drag
release, so it’s watching for that and making sure you’re lined up.”

A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting.
</item 2>

<item 3>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
ET, 7 p.m. CT in Chicago.

The game is being played at PPG Paints Arena in Pittsburgh, Pa.

MORE: Ranking the NHL's top 50 players for the 2023-24 season

What channel is Blackhawks vs. Penguins on today?

TV Channel (U.S): ESPN

ESPN TV Channel (Canada): Sportsnet, TVA Sports

Sportsnet, TVA Sports Live stream (U.S.): ESPN+, Fubo

ESPN+, Fubo Live stream (Canada): Sportsnet+

ESPN has coverage of the Blackhawks vs. Penguins game on Tuesday evening. The Pittsburgh-Chicago contest is one of three games that the network has on opening
night.

ESPN's "A-team" will be on the call for Bedard's debut, as Sean McDonough (play-by-play), Ray Ferraro (analyst) and Emily Kaplan (reporter) will be the
broadcast team for Blackhawks vs. Penguins.
</item 3>

<item 4>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
What is it, really, about Connor Bedard’s shot — a shot that, though he’s just 18 years old, has for years been talked about as if he has patented it — that
makes it so dangerous and unique?

What’s the nerdy science of it, from his feet up to his knees, hips, hands and head? How does he prep, shape and let it go? What does it look like to goalies,
and to professional shooting coaches, those who’ve taught him — or, more accurately, watched him after he taught himself — and had to stop him? The Athletic
spoke with five NHL shooting and skills coaches, his teammates, past opponents and him to try to take apart, piece by piece, “The Bedard.”

Advertisement

Their intel and stories help frame the NHL’s brightest young star’s signature skill.

If there’s one thing Tim Turk knows, it’s shooting.
</item 4>

<item 5>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Bedard wasn’t so reliant on teammates on those goals, either. They were more about the work he’s putting in on the other side of the puck.

“Even in practice, we’ve talked to him about maybe tracking harder and attacking pucks on the forecheck and showing him a couple clips,” Richardson said. “Ten,
12 games in, he’s really figuring things out and realizes — why sit back and let things come to him? Go get it.”

None of this is surprising to anyone involved. Bedard doesn’t show it if he impresses himself. And as grateful as the Blackhawks are to have drafted Bedard,
this is what they expected. This is why teams lined up to take losses last season.

“It’s not the NHL that he was playing in the last few years,” Davidson said, “but the level of performance and how he could control a game and dominate a game
in junior — it indicated he was going to be able to do that at some point in the NHL.
</item 5>

<item 6>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Rust gets the Penguins their first of the season 🚨 pic.twitter.com/RKNyaJT0LD — B/R Open Ice (@BR_OpenIce) October 11, 2023

8:28 p.m. — It's a strong showing from the top PP unit for the Blackhawks, but Bedard and Co. can't capitalize on the man advantage. Chicago had four shots on
goal, including a Bedard one-timer that was stopped by the pad of Mrazek.

8:26 p.m. — Penguins forward Lars Eller is called for goalie interference, as he clips the leg of Petr Mrazek in the crease. The Blackhawks head to the first
power play of the game about five minutes into the period.

8:23 p.m. — Through the first four minutes, it's been a back-and-forth affair. Neither team have been able to sustain much pressure. Most of Chicago's shot
attempts have been from the outside, making it easy on Tristan Jarry.
</item 6>

<item 7>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“So many players pull the puck in but can’t get the puck to release from under their body. His footwork is so underrated in that aspect. If you watch his front
leg, at times it’s literally in the air at release — most have their back leg in the air at release. When he transfers his weight, he actually clears space for
his hands to get tighter because he’s not afraid to actually lunge into the shot.”

A fourth shooting coach, who hasn’t worked with Bedard and also requested anonymity because he works for another NHL club, expressed a little more hesitancy.

Advertisement

“I’m in the minority I’m sure, but I am very curious to see how well his trademark shooting style works at this level,” the coach said. “He really likes to pull
pucks right into his feet.
</item 7>

<item 8>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Bedard was as advertised in his NHL debut. The Chicago rookie led all players on his club in shots on goal, recording five in his first game on 11 shot
attempts. His 21:29 time on ice was second to only Seth Jones, as Chicago head coach Luke Richardson did not hold the 18-year-old back. He did struggle in the
faceoff dot, winning just two of 13 draws, but it was a strong first impression at the NHL level.

For the Penguins, the third period struggles from last season have seemingly carried over to start this year. Last year, Pittsburgh finished tied for the most
losses after taking a lead into the third period. The crew followed that trend on Tuesday night, letting a 2-1 lead to start the final frame slip away.

Bedard and the Blackhawks are back in action on national TV tomorrow, as Chicago heads to Boston for a Wednesday night date with the Bruins.

MORE: Watch Blackhawks vs.
</item 8>

<item 9>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
While almost all players point their toes at the net, set the puck at a right angle, load their shot and “drag it in and then snap it off,” Turk says, on a line
of 40 to 50 degrees behind their body or at least level with their heels, Bedard brought the puck in “completely lateral.”

Instead of drawing the puck in at a 45-degree angle, he drew it at 180 degrees “right across to the plane of his spine.” Sometimes it even came in at a negative
angle. It also happened in a smaller space than it does for other shooters. That lateral inward pull was “condensed — it’s compacted.”

Turk says that Vladimir Tarasenko, whom Turk calls one of the NHL’s quickest shooters, takes 18 inches to pull the puck in. Bedard did it all in 12 square
inches — a one-foot box.

Turk’s reaction?
</item 9>

<item 10>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Korchinski was the No. 7 overall pick in the 2022 draft, one of three first-round selections by Chicago that year. He played his junior career with the Seattle
Thunderbirds of the WHL.

7:36 p.m. — You want to talk confidence? How about going without a bucket in warmups? Bedard is rocking the no-helmet look ahead of his NHL debut.

Connor Bedard going with the no-helmet look during pregame warmups. #Blackhawks pic.twitter.com/qrr93vlMsh — Charlie Roumeliotis (@CRoumeliotis) October 10,
2023

7:33 p.m. — A stick may be important when playing the game of hockey.

Note to self:

Don't forget your stick ahead of your NHL debut. pic.twitter.com/guzVprRXfY — Sportsnet (@Sportsnet) October 10, 2023

7:02 p.m.
</item 10>

<item 11>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson with the moves 👀#LetsGoPens | #NHLFaceoff pic.twitter.com/GCWVIGnAS2 — Hockey Daily 365 l NHL Highlights & News (@HockeyDaily365) October 11, 2023

8:41 p.m. — Pittsburgh heads to its first power play of the game, as Wyatt Kaiser trips up Malkin in the neutral zone. This will be the first time seeing Erik
Karlsson running the power play for the Penguins, taking Letang's place on the top unit with Crosby, Malkin, Guentzel and Rickard Rakell.

8:36 p.m. — Evgeni Malkin makes a beautiful saucer pass to Jake Guentzel cutting the net, but the Penguins winger could not elevate the puck over the shoulder
of Mrazek. At the other end, Taylor Raddysh took a feed from behind the net, but put it right in the midsection of Jarry.

8:34 p.m.
</item 11>

<item 12>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
📺: ESPN, ESPN+ pic.twitter.com/Qsk9Jgfp1c — Pittsburgh Penguins (@penguins) October 10, 2023

#Blackhawks Opening Night lineup vs. Penguins:

Hall-Bedard-Donato

T. Johnson-Reichel-Raddysh

Katchouk-Guttman-Athanasiou

Perry-Dickinson-Foligno



Vlasic-Jones

Korchinski-Murphy

Tinordi-Kaiser



Mrazek

Söderblom



Extras: Entwistle, R. Johnson, Zaitsev — Charlie Roumeliotis (@CRoumeliotis) October 10, 2023

7:50 p.m. — Bedard isn't the only rookie making their debut tonight. Blackhawks defenseman Kevin Korchinski is playing in his first NHL game tonight. Korchinski
was the No.
</item 12>

<item 13>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
You just never how quickly.

“It’s a big adjustment, especially when you’re playing against men and being focused on by the other team every single night. I’m excited for him, but I don’t
necessarily feel like it’s a huge surprise how quickly he’s started to really find his groove at the NHL level. But it’s exciting to watch.”

(Top photo: Ethan Miller / Getty Images)
</item 13>

<item 14>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
But just because injuries might take some of the juice out of these games doesn't mean that SuperDraft isn't giving us plenty of reasons to care. There are some
snazzy looking props out there for us to make our bottom line look at lot better than it did earlier this morning.

SuperDraft users can win 3X the amount wagered if their two-legged player prop parlay hits. If all four of the props below hit, users will win 10x their entry.
In other words, a $20 entry fee will yield a $200 payout. the more legs you add to your player prop parlay, the larger your potential payout becomes, but as the
long odds indicate, completing a clean sweep is challenging.

Below, we'll break down four of our favorite player props for Tuesday night's NHL slate. Users don't necessarily have to parlay all four picks together, though.
</item 14>

<item 15>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Daniel Sprong, RW, Red Wings @ Rangers — OVER 2.5 shots on goal

Despite not being featured in the top six, Sprong is holding down the fort on the third line, averaging five shot attempts per game through the first 12. He's
hit the over in five straight, with 18 SOG compared to just eight combined for his two linemates. Sprong does see some work on the second power play unit, but
is clearly willing to fire a lot of pucks at the net with his shoot-first mentality on the third line. With the Rangers still figuring out how to deal with the
loss of Fox, there should be some opportunities there for Sprong to find at least three shots tonight.

3. Jesper Bratt, LW, Devils @ Avalanche — UNDER 0.5 assists

Bratt comes into Tuesday night's action with seven goals and 11 assists in 11 games.
</item 15>

<item 16>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“Well who the f— are these guys?” Turk finally asked him.

“Turky, it’s Connor Bedard,” Landry said.

“Who’s Connor Bedard?” Turk asked.

After the second session that same week, Turk didn’t work with Bedard again until a stick manufacturer tried to pitch Bedard, who asked if Turk could be there
when he test-drove the sticks.

By then, Turk knew the Bedard name and came away from another session with another takeaway, picking up on Bedard’s eye contact and ability to change his mind
mid-shot.

“What makes him unique is that he can be selective with it,” Turk said.

When he looks back on that first blind introduction, Turk laughs.

“You know, most, when you’re in a non-pressure, non-stressful situation, you’ll play around with the puck a lot and then you’ll take your shot because I’m
telling them to take their time,” he said.
</item 16>

<item 17>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The winger came firing out of the gate, but the outlook has changed for the Devils in recent days after losing the superstar center Hughes to injury. Hughes
went down after slamming into the boards against the Blues, forcing New Jersey to change things on the fly. The changes have directly impacted Bratt, who was
featured on the top line alongside Hughes. With fellow center Hischier also out with an injury, leaving the Devils fairly weak down the middle and trusting
Michael McLeod to hold down the fort in between Bratt and Tyler Toffoli. As a result, Bratt has been held without an assist in each of the last two games for a
team that's missing a key contributor on offense. And it'll need him to look to score rather than play the role of playmaker. Until the Devils figure things
out, we can't trust this team to do anything positive in the offensive zone.

Win BIG with SuperDraft!
</item 17>

<item 18>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The Wild will certainly get a boost in the standings once Fleury and Gustavsson improve their play between the pipes, a likely scenario given their stellar
performance last season. However, they again have the misfortune of being in quite a competitive division. The Avalanche and Stars occupy first and second place
as expected powerhouses, while the Blues, Predators and Coyotes all sit at or above the .500 mark and, especially in Nashville’s case, have posted decent
underlying numbers.

Hynes’ main job is simply to restore confidence in his group. This roster has the ability to play better — especially Kaprizov, who, despite the okay point
production, has looked like a complete non-factor on some nights. The team also hopes a significant change behind the bench can boost their league-worst penalty
kill, which is operating at just 66.7%. Penalty killing was an area of strength for Nashville last season, who finished tied for fourth in the NHL with an 82.6%
success rate.
</item 18>

<item 19>
Who’s the next Gawn, Grundy or Dusty? AFL’s talent guru names top 40 draft prospects and compares them to current stars (The Roar | Sports Writers Blog,
2023-11-15T23:48:27+00:00)
Weapon: Consistent high-level performances

AFL player comparison: Andrew Brayshaw (Fremantle)

Ari Schoenmaker

Pronunciation: Show-en-maker

State: Tasmania

State League Club: Tasmania Devils

Community Club: Launceston

Date of Birth: 07/01/05

Height: 194cm

Left-footed tall defender who reads the game exceptionally well, is a good interceptor and prodigious kick of the ball. Had a very impressive back half of the
year, which included being selected for the Allies in their final two matches of the National Championships. He also showed the ability to play on the wing and
averaged 27 disposals at 75 percent efficiency as well as 5.9 marks and 9.6 rebound 50s in his eight matches with the Tasmania Devils in the Coates Talent
League to be one of their most impressive performers.
</item 19>

<item 20>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
What do the Giants’ and Vikings’ ownerships think?

“The best first year can be that 7-10 season where you show some promise,” a veteran coach with experience on a half-dozen teams said. “The Giants won a bunch
of games early, people were heaping praise on the head coach, people get the expectation of what the quarterback can be. But oftentimes, you know different
internally because you know the house you are living in. You are inside the walls every single day and you know where the roof leaks, where the foundation is
cracked.”

Buffalo managed such a situation well. The Bills surprisingly reached the playoffs in their first season under Sean McDermott, a euphoric moment after a 17-year
playoff drought. They went 6-10 the next year, then began a steady rise behind Josh Allen and a top defense. There isn’t a Josh Allen on the Giants or Vikings.
</item 20>

<item 21>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
First game, first point, first win.

Connor Bedard's debut did not disappoint on Tuesday, as the Chicago rookie helped the Blackhawks to a 4-2 win over the Penguins on opening night.

The Blackhawks rallied from a 2-0 deficit, scoring four unanswered goals to pick up the victory in Pittsburgh.

Bryan Rust opened the scoring for the Penguins in the first period, and a Sidney Crosby goal in the second doubled the Pittsburgh lead. However, Bedard helped
set up a Ryan Donato goal in the middle frame to get the Blackhawks on the board, picking up his first NHL point in the process.

In the third period, Cole Guttman potted the equalizer for Chicago early in the frame. With less than five minutes left in regulation, Jason Dickinson fired
home a shot from the slot that gave Chicago its first lead of the night. A Nick Foligno empty net goal sealed the deal for the Blackhawks.

Bedard was as advertised in his NHL debut.
</item 21>

<item 22>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Ryan Donato slips home a rebound from the top of the crease to put Chicago on the board, and Bedard has his first NHL point, picking up a secondary
assist on the tally. Alex Vlasic walks in off the wall and puts a low shot on Jarry that is saved, but Donato pulls the rebound around and by Jarry's leg with
4:23 left in the second.

🚨first NHL assist for Connor Bedard. the full sequence starting with his zone entry pic.twitter.com/JqJMhchiTN — Shayna (@hayyyshayyy) October 11, 2023

Ryan Donato puts home the juicy rebound to give Connor Bedard his first NHL point!#Blackhawks | #NHLFaceoff pic.twitter.com/JbjVPvHVLE — Hockey Daily 365 l NHL
Highlights & News (@HockeyDaily365) October 11, 2023

9:46 p.m.
</item 22>

<item 23>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
It's 2-2 with 9:55 left in regulation.

Tie game with less than half the period to go 👀 pic.twitter.com/uqB1Z2Bnol — B/R Open Ice (@BR_OpenIce) October 11, 2023

10:24 p.m. — The Blackhawks can't take advantage of a power play again, dropping to 0 for 4 tonight on the PP. Pittsburgh remains ahead 2-1 with 13:43 left in
the period.

10:20 p.m. — Pittsburgh kills off the Nieto penalty, but seconds after going back to even strength, Letang picks up a cross-checking minor. Chicago goes right
back to the man advantage.

10:19 p.m. — Corey Perry had not just one, but two point-blank chances in front of the Pittsburgh net, but he put the puck right into the midsection of Jarry
both times.

10:17 p.m.
</item 23>

<item 24>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
A number of Chicago Blackhawks over the last few decades liked to put in extra time shooting. Patrick Kane would stay out with the young guys, including Alex
DeBrincat, well after practice to play shooting games. But even they had a limit.

Bedard doesn’t have much of a limit. It’s common for him to be on the ice an hour after practice taking shots from all over the ice. He even got a few other
rookies, especially defenseman Kevin Korchinski, to join him on a regular basis. As the media waits for Bedard at his dressing room stall for interviews, he
shoots and shoots and shoots.

Advertisement

Blackhawks coach Luke Richardson has been around long enough to see that in a few other special players.

“I know Jaromir Jagr did this for a long time,” Richardson said. “Even at the end of his career — and he skated around and he had the weights around the bottom
of his ankles, practiced full practices like that. It’s happened.
</item 24>

<item 25>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Users don't necessarily have to parlay all four picks together, though. Instead, you could make separate two-legged parlays to increase your win probability or
attempt to hit it big if all four picks come through.

Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

Best NHL prop bets today: SuperDraft player prop picks for Wednesday

*10x potential payout if users parlay all four correct picks

1. Artemi Panarin, LW, Rangers vs. Red Wings — OVER 3 shots on goal

Panarin has registered at least three shots on goal in nine of 11 games this season, going over three in six of those contests. The winger has been up to his
usual antics in the new season, but is also seeing some extra ice time under new head coach Peter Laviolette.
</item 25>

<item 26>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“I can tell you firsthand it didn’t happen by accident. Connor’s worked on this since he was little,” said Quinn, an NHL skills coach who has worked with Bedard
each summer for several years.

If there’s another, it’s how hard Bedard’s shot is to defend because he doesn’t show you he’s preparing for it.

“As replicated in so many other areas of Connor’s game, it’s the deception and elite multi-tasking that catches opponents off guard,” Quinn said. “Connor’s
ability to create deception and change the shot angle at top speed is like very few I’ve ever seen. The multi-tasking involved with attempting these shots at
top speed is far beyond most player’s capabilities.”

Another shooting coach, who requested anonymity for this story because he works for another NHL club, pointed to Bedard’s hands and legs.

“What’s really interesting is how high and left he can get his top hand,” the coach said.
</item 26>

<item 27>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Tuesday night's NHL national games feature an Original Six matchup at Madison Square Garden in addition to a battle between the team that left and the team that
took its place. The Rangers and Red Wings will face off at the world's most famous arena as both squads have jumped out to hot starts in the new season. Dylan
Larkin is a game-time decision for the Motor City while injuries to the Blueshirts' Adam Fox, Filip Chytil and Igor Shesterkin are making sure some of this
matchup's biggest stars will not be watching in uniform. Regardless, there is still plenty of star power on Broadway in what should feature plenty of offense.

In the late game, the Devils return to Colorado -- the place they left in order to move to New Jersey -- for a showdown with the Avalanche. Again, injuries are
a big part of the storyline here, thanks to Jack Hughes and Nico Hischier out until further notice.
</item 27>

<item 28>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
However, the Blackhawks are challenging the play for offside. Marcus Pettersson had to reach for the puck along the blue line and it may have come out of the
zone.

9:27 p.m. — Jarry keeps the shutout going with two real strong saves on Johnson. A shot from the point was tipped by an unmanned Johnson at the top of the
crease. When that didn't go, he got a second whack at the loose puck, but Jarry stuck out the leg to make the stop.

9:23 p.m. — Reilly Smith, who was traded to the Penguins this offseason from the Golden Knights, is denied by the glove of Mrazek off the rush. Smith had a puck
trickle to him on the right side, but he shot it right into the mitt of the Blackhawks goaltender.
</item 28>

<item 29>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— GOAL! Perry finds Foligno for the empty net goal. The veteran line gave the Blackhawks the lead, and then the unit ices the game with 1:27 to go.

10:47 p.m. — Penguins head coach Mike Sullivan pulls Jarry for the extra attacker with over two minutes to go.

Blackhawks 3, Penguins 2

10:40 p.m. — GOAL! Chicago takes its first lead of the game thanks to Jason Dickinson. Nick Foligno drops a pass on a zone entry to Perry, who tries to get a
shot on net, but it deflects to the middle and Dickinson snaps it home. The Penguins, who struggled with third period leads last year, have given up three
unanswered and find themselves down one with 4:33 to go.
</item 29>

<item 30>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
pic.twitter.com/6TWSCfXUlR — Steven Ellis (@SEllisHockey) October 11, 2023

8:47 p.m. — Noel Acciari and Tyler Johnson collide in the neutral zone, and Johnson is left leaking from the bridge of his nose. Initially Acciari is assessed a
double minor for high-sticking, but after reviewing the play, no penalty was killed. The officials deemed that the stick was not above the shoulders, therefore,
no infraction.

8:46 p.m. — The Blackhawks kill off the Kaiser minor. Pittsburgh managed to get three shots on Mrazek, but could not build on its lead.

8:43 p.m. — Karlsson dances by Cole Guttman at the blue line, walks into the slot and lets one rip, but Mrazek stares it down the whole way, making the glove
stop on the Penguins defenseman.
</item 30>

<item 31>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
The Rangers head into this matchup with a makeshift lineup thanks to those injuries that is forcing them to adjust. The absences of Fox and Chytil are a big
deal, however, as Panarin loses Fox, who quarterbacks the power play, and Chytil, who centered his line at even strength. Panarin is still all over the ice for
the Blueshirts though and is even seeing 18.8 percent of his shifts start in the offensive zone, a welcome sight after the last two seasons with Gerard Gallant
behind the bench. The Red Wings have allowed the 11th most SOG this season (371) and the fourth most power play opportunities (46). If the Rangers can play with
the man advantage, Panarin should hit this number with ease, but he's enough of a factor at even strength that we like the OVER here anyway.

MORE SUPERDRAFT DFS & FANTASY PROPS: How to play SuperDraft

2.
</item 31>

<item 32>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
If there’s one thing Tim Turk knows, it’s shooting. A self-described “shooting and scoring coach,” Turk has more than two decades of experience working with NHL
stars, national federations and half a dozen NHL teams.

Turk had no idea who Bedard was when he fielded a call from agent Greg Landry of Newport Sports asking him what he was doing tomorrow.

“Tomorrow?” he said. “I’m booked up a year in advance.”

“We have these two players coming in from out west and we want you to see them,” Landry said.

“Listen, I’m booked up,” Turk said again.

“You have to see them,” Landry insisted.

After moving around his schedule, Turk found himself at Gary Roberts’ personal rink in Uxbridge, Ontario.

“Hey guys, I’m Tim Turk,” he said as he stepped into the small locker room.

“My name’s Nate,” said the redhead, extending a hand.

“My name’s Connor,” said the other, standing shorter and extending his.
</item 32>

<item 33>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Obviously, the goalies are good, you got to hit your spot.”

Blackhawks goalie Arvid Söderblom would certainly call Bedard a shooter. Söderblom and Petr Mrazek have faced more shots from Bedard than any NHL goalies this
season. The challenge for them is that Bedard doesn’t just shoot one way. He’s unpredictable.

“When you have that type of shot, quick release, moves his body and shifts angles on the puck, it makes it harder on the goalies,” Söderblom said. “You just saw
the two goals (on Sunday against the Florida Panthers), both were quick releases. He finds that open net. He’s a pure goal scorer. It’s fun to have on the team
and face him every day and see him take steps, too.”

While goalies are trying to figure out Bedard, he’s been putting in the time to do the same with them. For one, he learned to utilize his teammates more.
</item 33>

<item 34>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
MORE: Watch Blackhawks vs. Penguins live on Fubo (free trial)

The Sporting News provided updates and highlights from Connor Bedard's NHL debut in the Blackhawks vs. Penguins game.

Blackhawks vs. Penguins score

1 2 3 F Blackhawks 0 1 3 4 Penguins 1 1 0 2

Blackhawks vs. Penguins live updates, highlights from Connor Bedard's NHL debut

All times Eastern

Final: Blackhawks 4, Penguins 2

10:50 p.m. — The Blackhawks score four unanswered goals to come from behind and take down the Penguins on opening night. Bedard was as advertised in his debut,
picking up his first NHL point in the process. The third period struggles continue over from last season for the Penguins.

10:48 p.m. — Jarry back to the Pittsburgh bench again for the extra attacker.

Blackhawks 4, Penguins 2

10:48 p.m. — GOAL!
</item 34>

<item 35>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
Injuries and goaltending have certainly played a factor in their poor play — captain Jared Spurgeon missed the first 13 games of the season, and Filip
Gustavsson’s and Marc-André Fleury’s combined .878 SV% is near the bottom of the league.

They’ve likely been unlucky, too. The Wild have controlled a slim majority of scoring chances and a strong majority of high-danger chances during 5-on-5 play, a
common theme for an organization that’s usually one of the more defensively stout in the league. But the culture around the team is evidently turning sour
quickly, Kirill Kaprizov is on pace for a career-low 26 goals and has a team-worst -10 rating, and most of their depth defensemen have been significant
liabilities. This roster is not all that different from the one that’s finished top-three in the Central Division for three straight seasons, and it’s certainly
one that still has playoff aspirations.

Minnesota has not won since Nov. 7 against the Islanders and is winless in seven straight.
</item 35>

<item 36>
Best NHL prop bets tonight: Artemi Panarin, Cale Makar highlight top parlay picks for Rangers-Red Wings, Devils-Avalanche (Sporting News,
2023-11-07T22:09:00+00:00)
Win BIG with SuperDraft! Get a free $10 deposit when you use promo code "TSN"!

4. Cale Makar, D, Avalanche vs. Devils — OVER 2.5 shots on goal

Over this number in each of the last two, fans will want to see Makar shoot the puck a little more after going under in five of the previous six before his
recent hot streak. Facing a Devils team that is reeling thanks to injuries should help Makar's cause with Colorado able to control the play in the Mile High
City. He's the team's top defenseman and a key piece to their power play. Barring something unforeseen, Makar will have plenty of opportunities to find three
shots before the 60 minutes on the clock run out. And with 5.9 shot attempts per game, it's only a matter of time before they start finding their way on goal.
</item 36>

<item 37>
Wild fire HC Dean Evason, hire John Hynes as replacement (Yardbarker, 2023-11-27T23:45:08+00:00)
The 59-year-old Evason began his coaching career with the WHL’s Calgary Hitmen as an assistant in 1998-99 after finishing out his playing career in Germany,
working his way up the junior hockey ranks over the next handful of seasons. His first NHL job came courtesy of an assistant role with the Capitals immediately
after the 2004-05 lockout, coaching Alex Ovechkin in his first seven NHL seasons.

After departing Washington in 2012, Evason spent the next several seasons working in the Predators organization as the head coach of their AHL affiliate, the
Milwaukee Admirals. There, he helped guide more than a few future NHL stalwarts — namely Mattias Ekholm, Ryan Ellis, Filip Forsberg, Calle Järnkrok, Viktor
Arvidsson, and Kevin Fiala, among many others. After joining the Wild, he earned a promotion to head coach within two seasons when the Wild fired Bruce Boudreau
late in the 2019-20 campaign.
</item 37>

<item 38>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
A strong showing so far from Bedard, who has not shied away from putting the puck on net. He had four shot attempts in the opening frame, with two on net. The
Blackhawks out-shot the Penguins 17-13, but Pittsburgh holds the lead.

8:51 p.m. — An excellent shift by Bedard nearly results in his first goal. Initially, Bedard shows off his elite release by zipping a shot from long-range, but
it's turned aside by Jarry. Later in the shift, the rookie gets two whacks at a loose puck in front of the crease, but he could not beat the Penguins
goaltender.

Bedard shows off the release, Jarry with the save pic.twitter.com/BKLDkQdpwD — Hockey Realm (@hockey_realm) October 11, 2023

Connor Bedard with his first good scoring chance.
</item 38>

<item 39>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Most of Chicago's shot attempts have been from the outside, making it easy on Tristan Jarry.

8:19 p.m. — It was an uneventful first shift for Bedard, as the Penguins controlled most of the play in the Blackhawks' zone.

8:17 p.m. — Bedard and Crosby square up for the opening face off, and Bedard's debut is underway!

Sidney Crosby 🆚 Connor Bedard on the draw to start this one off.



What a moment. pic.twitter.com/4BPboh8jdp — Sportsnet (@Sportsnet) October 11, 2023

Pregame

7:58 p.m. — Here is how the two sides are lining up for the season opener.

Almost time for hockey!
</item 39>

<item 40>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“My name’s Connor,” said the other, standing shorter and extending his.

“All right, well we’re just going to go out and I’m going to do an evaluation on you and do some shooting and have some fun, and I’ll make suggestions but I’m
not here to change anything,” Turk said, giving the two boys his usual spiel and thinking nothing of it.

That evaluation is what he calls an “NHL protocol observation assessment,” which runs new clients through specific drills that allow him to visualize their
shooting actions from a technical standpoint (he calls each player’s shot “like a fingerprint”).

That starts with Turk throwing some pucks down in the slot and simply asking them to shoot them stationary so he can watch their body formation, hip and
shoulder positions, spine angles and puck preparation and positioning, paying particular attention to where the puck starts in their stance versus where it
finishes.

Advertisement

On that day, Nate (Danielson, the Red Wings’ 2023 No.
</item 40>

<item 41>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
“He really likes to pull pucks right into his feet. I really like his mechanics with his hands off his body and whippy stick — but I feel this habit makes it
easier for elite level ‘D’ to get stick on puck or in the lanes to block shots. Finding space and time to let that shot go and get clean looks, I think, will be
an adjustment.”

For goalies, it’s the angle change that gets them.

Moose Jaw Warriors goalie Jackson Unger, whom Bedard scored on last year, said the toe-drag release, in particular, is a challenge.

“He changes the angle so quickly that, as a goalie, you have to adjust to it, but when he does it so fast, it’s easier said than done,” Unger said. “It’s a lot
of different angles he can give you.”
</item 41>

<item 42>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
The Eagles come to mind after falling 20-14 to the Jets.

Most would agree teams have gotten smarter in their handling of fourth-down situations, going for it when it makes sense, instead of punting almost out of
habit. Making the aggressive play to win can absolutely be the smart play, regardless of results. It’s progress when teams seriously wrestle with these
questions, not just on fourth down, but in all aspects of their strategy.

But when the Eagles tossed a killer interception on third-and-9 from their own 46-yard line while protecting a 14-12 lead against the Jets with two minutes left
in the fourth quarter, a longtime NFL exec reached out with a question: “Have we come too far with coaches wanting to ‘win’ the game instead of sometimes being
more traditional?”

His point: Philadelphia could have run the ball on third-and-9, then punted from around midfield.
</item 42>

<item 43>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
Turk’s reaction?

Advertisement

“I was like holy f—,” he said, spelling out the whole profanity, “because the only other person that I know who does it quite like that is No. 34 in Toronto
(Auston Matthews), and he still doesn’t bring it in as lateral as Bedzy does.”

After the stationary shooting, they got into some motion drills.

Every time Turk introduced a new formation and shot type for Bedard, his application was “simplistic, and he made it look easy.”

View this post on Instagram A post shared by Tim Turk Hockey (@timturkhockey)

When it was over, both kids said, “This was fun, thanks coach Turky,” and Turk got in his car and drove home.

The next day, he got another call from Landry asking him what he was doing Thursday because the two boys had asked if they could have him back for another
session.

“Well who the f— are these guys?” Turk finally asked him.
</item 43>

<item 44>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
— If it feels like Sidney Crosby, Kris Letang and Evgeni Malkin have been in Pittsburgh forever, that's because it had. The trio are set to surpass Yankees
legend Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio of teammates in North American sports history.

Ready for the Big Three to take the ice for their 18th season together.



Sidney Crosby, Evgeni Malkin and Kris Letang are set to surpass the New York Yankees’ Derek Jeter, Mariano Rivera and Jorge Posada as the longest-tenured trio
of teammates in NHL, MLB, NBA and NFL history. pic.twitter.com/bvcWZOw7kO — Pens Inside Scoop (@PensInsideScoop) October 10, 2023

6:45 p.m. — All the buzz has been about Bedard leading up the matchup, but there's a new Penguin on the roster.
</item 44>

<item 45>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
10:17 p.m. — The Blackhawks are going to their third power play of the night as Matt Nieto is called for hooking just 2:28 into the third.

10:14 p.m. — The third period is a go from Pittsburgh with the Penguins holding onto a one-goal lead.

End of the second period: Penguins 2, Blackhawks 1

9:56 p.m. — Both teams get on the board in the middle frame, but it's the Penguins that remain ahead going into the final period. Bedard looks to be improving
every shift and he picks up his first NHL point on the Donato goal.

9:54 p.m. — An awful line change by the Blackhawks allows Malkin to send Smith on a breakaway, but Mrazek comes up with the stop.

Penguins 2, Blackhawks 1

9:47 p.m. — GOAL!
</item 45>

<item 46>
Blackhawks vs. Penguins live score, updates, highlights from Connor Bedard's NHL debut (Sporting News, 2023-10-10T22:30:03+00:00)
Erik Karlsson, who won his third Norris Trophy last season, was brought in this summer in a trade with the Sharks. It gives the Penguins a talented duo on the
blue line of Kris Letang and Karlsson, although the two likely won't be paired together.

Actually, this is exactly what we’ve been waiting for. pic.twitter.com/0ZJA4YRs6v — Pittsburgh Penguins (@penguins) October 10, 2023

6:30 p.m. — Connor Bedard is in the house for his NHL debut. It's the most anticipated first game for a rookie since Auston Matthews' debut in 2016.

Blackhawks vs. Penguins start time

Date: Tuesday, Oct. 11

Tuesday, Oct. 11 Time: 8 p.m. ET (7 p.m. CT)

The Blackhawks and Penguins are set for puck drop at 8 p.m. ET, 7 p.m. CT in Chicago.
</item 46>

<item 47>
N.F.L. Executives on the Biggest Questions Facing Super Bowl Contenders (The New York Times, 2023-10-16T23:58:53+00:00)
These game plans were conservative enough to signal a potential in-season quarterback change. Denver faces Green Bay and Kansas City over the next two weeks
before heading into its Week 9 bye.

5. You play to win the game, but can you win too quickly? The Giants and Vikings are test cases.

The New York Giants and Minnesota Vikings exceeded expectations under first-year head coaches in 2022. Both reached the playoffs. Both are struggling in their
second seasons, facing difficult questions about the future. Would they have been better off enjoying only moderate success early? That could depend on how well
the football leadership has communicated with ownership.

Every head coach and general manager wants to win as much as possible, but the goal is to build for the long term, which can require time to do things with
sustainability in mind. Bad teams that win right away can reset expectations for fans and ownership, making it less practical to take one step backward in the
short term for a better shot at taking two steps forward down the line.
</item 47>

<item 48>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
9 pick) went first, and Turk could tell he was a talented and hard-working player with a good shot and real promise.

Then, on Connor’s turn, Turk placed him in a position and watched how he got the puck ready to shoot. Only Connor didn’t stickhandle and prepare for the shot,
he just took the puck and shot it right away.

After watching him run through the same number of shots as Nate did, Turk spoke for the first time.

“Timeout one second, just give me a second here, I’m just trying to see what you’re doing so I’m going to have you take an extra few shots so that I can analyze
where the puck’s starting and where it’s finishing,” he said.

Another round later, he knew that this Connor kid, whose last name he still didn’t know and who was then 16, was very different from even his high-end peers
simply by his proficiency in shooting from a “frontal position.”
</item 48>

<item 49>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
For one, he learned to utilize his teammates more. He accumulated plenty of shots on net early in the season, but they were often from distance and more
individualistic chances. Then, he scored his first NHL goal on a wraparound and his next three goals off passes from teammates.

Advertisement

“Look, I’m sure he’ll figure out the one-on-one and how to score that way or create chances that way,” Blackhawks general manager Kyle Davidson said. “There is
also an element of working off your teammates. You are in the NHL, so everyone is a pretty good player and you’re able to create using more than just yourself.
I think he’s figured that out.”

Bedard found more success on the Blackhawks’ last road trip. He left Chicago with five goals in 11 games and returned from Florida with nine goals in 13 games,
scoring twice against the Tampa Bay Lightning and again against the Panthers.

Bedard wasn’t so reliant on teammates on those goals, either.
</item 49>

<item 50>
He’s Hockey’s Brightest Young Star. This Is What Makes His Shot So Special. (The New York Times, 2023-11-17T21:47:19+00:00)
It’s happened. That’s their life. They love it so much and they have to be out there. That’s where they feel comfortable and don’t get tired. Hopefully, they go
home and have a nap in the afternoon and not watch ‘Young and the Restless.’ Take it easy and get ready for the next day. It does add up when you put the hours
of the week in.”

Bedard has downplayed his lengthy post-practice sessions. To him, it’s not like he has weights on his ankles. He’s just shooting, nothing too strenuous.

For someone who shoots as much as Bedard does, you’d think he would gladly accept being called a shooter. He doesn’t.

“I say it a lot: I don’t feel like I’m a shooter,” he said. “I’m just trying to make the right play. If the shot’s there, I’ll take it. Obviously, the goalies
are good, you got to hit your spot.”
</item 50>
=== END ITEM id=7c233b9758b3eb02 ===
