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

This file has 1 items.

=== ITEM id=c50c1179b63deb61 ===
Gold sentences and facts:
Sentence 0: When he entered free agency, a dozen teams lined up, curious to see if they could meet his eye.
  Fact 0.0: A player entered free agency.
  Fact 0.1: A dozen teams lined up when that player entered free agency.
  Fact 0.2: The teams were curious to see if they could meet that player's eye.
Sentence 1: As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and determining the fate of those around me.
  Fact 1.0: The speaker is the player.
  Fact 1.1: The player explores a system of the Imperium.
  Fact 1.2: The player has a loyal crew of companions.
  Fact 1.3: The player makes impactful choices.
  Fact 1.4: The player determines the fate of those around the player.

Retrieved items:
<item 1>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Ana Diaz

49. Lies of P

Developer: Neowiz Games

Where to play: Mac, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Yes, Lies of P is a Dark Souls mixed with Pinocchio, and that’s a questionable elevator pitch from the outset.

In the years leading up to Lies of P, “Pinocchiosouls” was more of a running joke than anything — this profane idea that you can take any world and slap some
Dark Souls into it to get people mildly interested. But once you’re in the game, eliminating bosses left and right with your sweet parry moves, you’ll quickly
find yourself entirely unbothered by how strange Lies of P initially seemed. And you’ll start telling your partner things like “I have to go back to Geppetto to
upgrade my puppet body” like it’s a perfectly normal task to assign yourself on a Tuesday afternoon.
</item 1>

<item 2>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Who will hand him that contract has been much more of a mystery. Little has been leaked about the top suitors for Ohtani beyond the usual suspects. Passan
reported that at least three teams initially believed to be in the running — the Mets, Rangers and Red Sox — are believed to be out now. The teams most likely
to still be in the mix are the Dodgers, Cubs, Blue Jays and Angels, with the Giants' status in the race for Ohtani unknown.

Star players could be traded

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

The free agency pool outside of Ohtani, Cody Bellinger, Blake Snell and Yoshinobu Yamamoto is a bit lacking in star power, but the trade market could ensure
several big-name players are on the move this offseason.
</item 2>

<item 3>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can chat with the companions between battles, learning more about their pasts. Most of them have deep and dark secrets I can uncover with a little time or
patience, and they have fascinating stories to tell. Abelard, my Senechal, is a guy who sucks morally but will back me to the absolute hilt. I grew to love
hearing about his days in the Imperial Navy, and he was the one guy I felt like I could trust. Augusta, a Sister of Battle, starts as a one-note zealot, but
cracks form in that facade when I learn about her past and doubts. Cassia and Pasqal both represent two sub-factions in 40K, and they have lots to share about
the Navigators and Tech-Priests.

My absolute favorite companion is Marazhai Aezyrraesh, a dark space elf who feeds off the suffering of sentient beings. He’s cruel, depraved, and an absolute
hoot.
</item 3>

<item 4>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
11

Source: Fabian Ardaya, The Athletic

According to The Athletic, Ohtani will defer a whopping $680 million of his $700 million contract from the Dodgers, receiving only $2 million per year.

While it's important to note the deal will still count as $46 million annually toward the luxury tax thresholds for Los Angeles, Ohtani is still giving the
organization much more flexibility than expected over the course of the deal.

The two-time MVP will receive $68 million per year, without interest, from 2034-43, making the present-day value of the contract much less than it originally
appeared.

MORE: Complete breakdown of Shohei Ohtani's Dodgers contract, by the minute

Giants reportedly meet with Yamamoto

Date: Monday, Dec. 11

Source: Buster Olney, ESPN

ESPN's Buster Olney reports the Giants met with Japanese ace Yoshinobu Yamamoto on Sunday, Dec. 10.
</item 4>

<item 5>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—CA

Related Viewfinder is puzzle game heaven

24. Humanity

Developer: tha

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Humanity looks like a modern art project, recalls memories of early PlayStation oddity, and comes (in part) from the team behind game of the future Rez
Infinite. But its most impressive feat is the way it gradually shifts genres, starting as a classic puzzle game, then taking on tower defense and shoot-’em-up
traits as it evolves into an all-out war. Think Braveheart, with a shiba inu general leading a troop of faceless, brainless low-polygon figures against a group
of angry, lightsaber-wielding, even lower-polygon foes. It’s the kind of design subtlety that stays in its lane yet builds on you, and before you realize it,
ends very differently than it began. Much like humanity itself? —ML

23.
</item 5>

<item 6>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
This was a year of upheaval in video games. The industry has shapeshifted over the past 12 months, and it’s not all due to Microsoft’s lengthy acquisition of
Activision, Blizzard and King. While Xbox executives were defending the legality of a $69 billion deal that would create the third-largest video game studio in
the world, smaller companies were firing staff and shutting down entire teams, even amid fervent collective-bargaining efforts. It’s been a wild ride.

In 2023, the main factors molding the video game landscape were consolidation, layoffs and unionization, with each of these phenomena feeding into each other.
This past year, the video game industry shrank, even as it grew financially.

Consolidation

When Microsoft's purchase of Activision-Blizzard-King was finally approved on October 12, 2023 after nearly two years of regulatory hurdles, it became the
world’s third-largest video game studio by revenue.
</item 6>

<item 7>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
What’s more, Larian had two years of player feedback with which to build its new masterpiece. That masterpiece launched in August, and it garnered the kind of
attention that catapults a game from niche interests to widespread acclaim.

Old CRPG fans finally came home, the Mass Effect generation has discovered a new kind of RPG to sink its teeth into, and D&D fans have found an exceptional
digital version of their beloved tabletop game to play alone or with their party in co-op.

Like a good DM, Baldur’s Gate 3 teases out story beats that feel personal to you, cleverly luring you into experiences that seem like they wouldn’t make sense
in anyone else’s campaign. Every act delivers quiet, memorable character moments and anxiety-inducing battles. All of these situations feel organic, allowing
Baldur’s Gate 3 to replicate, perhaps as closely as a video game can, Dungeons & Dragons’ best feature: the intoxicating sense that anything could happen at any
moment.
</item 7>

<item 8>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game also sheds its predecessor’s odious Spider-Cop bits in favor of establishing a deeper connection between the Spider-Mans and their city, giving the
game room to breathe while further immersing you in its heroes’ worlds and struggles.

And then there’s the gameplay. For my money, there are few experiences more enjoyable in gaming than swinging around New York City with Spider-Man. It was fun
in the early 2000s, and it’s still fun now. And with a bigger map to explore (and wings to fly, if you so choose to use them), there’s basically no limit to the
fun you can get up to. The deployment of two protagonists is seamless — Peter and Miles play differently, bringing their own stories and desires to the table —
and switching between them is effortless. It’s also the rare open-world game that doesn’t feel bloated with missions and side quests, instead leaving you
wanting more Spider adventures. —PV

7.
</item 8>

<item 9>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
You can even wear them as mourning jewelry.

This Pathfinder 2nd edition-compatible setting has been as enchantingly addictive as the fictional ichor that propelled Ourichor into the renaissance era that
the project is named after. While there are only two exquisitely illustrated supplements on creatures and equipment available currently, I recommend those with
a love of the unusual to keep an eye out for more in the years to come, such as the upcoming Paragons of Luminant Age that introduces NPCs and explores their
motives through the four moons.

Marvel Multiverse Role-playing Game

As a role-player, I’ve spent a lot of time looking for a good lightweight superhero RPG. I’ve tried quite a few but had yet to find one that really hit the mark
for me until I tried Marvel Multiverse Role-playing Game. I have to say, I was initially dubious about this game but it has blown me out of the water.
</item 9>

<item 10>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 10>

<item 11>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
There are so many puzzles in this game — there’s even a puzzle metagame, spread across the game’s gorgeous map — and they’re each excellent, teaching you new
concepts before refracting them and forcing you to think differently. —Nicole Clark

Related November games you might have missed

44. Fading Afternoon

Developer: yeo

Where to play: Windows PC

Video games demand an account for your time. Some video games track the minutes and hours you’ve spent on screen. Others impose strict limits on how long you
play, or how much you can do. But all video games constantly ask: How will you spend your time? Seiji Maruyama doesn’t know how much time he has left. Fresh out
of prison, he’s not a young man anymore, but he doesn’t want the streets to know. So he returns to his life as a yakuza heavy, hoping to make his mark again.

Fading Afternoon ends when Seiji’s time runs out.
</item 11>

<item 12>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
For me, Project ECCO found that balance.

A story of time travel and cosmic horror, Project ECCO is easy to get into and relatively quick to play (three to 10 hours per game). Between the hundreds of
narrative prompts, unlocking various time travel “devices” (coins, dice, tarot cards, etc.), and the absolutely stunning layout, this is one of my favorite
TTRPGs, period.

Played in the pages of a yearly planner, Project ECCO hits every beat of a good time-travel story: time loops and timeline divergences, a shady and controlling
Time Travel Agency, and an unknowable cosmic Entity consuming spacetime. As you play, you (literally) burn through the days of your planner, creating an
artifact, marked and changed by the telling of your agent’s story.

For an example of gameplay and a masterfully produced actual play, listen to the Project ECCO miniseason of My First Dungeon with game designer Elliot Davis.
</item 12>

<item 13>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The game is utterly devoted to giving the players the feeling of playing a hero in the Marvel universe with tons of cool powers and power categories based on
key Marvel heroes like Spider-Man and the Hulk. The system is easy, streamlined, and really carries the four-color comic book flavor. Probably one of the
coolest aspects of the game is that while there is an exhaustive list of Marvel heroes and villains at the back of the book, you can create your own hero with
their own power set and use that “bestiary” of heroic and villainous figures as allies and enemies in your own heroic adventures. For those of us who’ve always
wanted to make their own X-Gene Mutant, Spirit of Vengeance, or Spidersona, this is a must-have book that will bring hours of enjoyment.

Masks of the Masks

Superhero comics, for all their many flaws, remain my life’s great passion.
</item 13>

<item 14>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Kansas City can use plenty of rotation help after a dismal 2023 season, so Lugo figures to be one of the Royals' top arms in 2024 and perhaps beyond.

Yamamoto to meet with Red Sox, Blue Jays

Date: Tuesday, Dec. 12

Source: Mark Feinsand, MLB.com

MLB.com's Mark Feinsand reports Japanese ace Yoshinobu Yamamoto will meet with both the Red Sox and Blue Jays in the coming days.

Yamamoto has already met with the Mets, Giants and Yankees, with the Dodgers also believed to be involved in the bidding. With both meetings still upcoming, the
report is an indication the Yamamoto sweepstakes aren't on the verge of a conclusion.

MORE: Ranking the best fits for Japanese star Yoshinobu Yamamoto

Dodgers formally announce Ohtani signing

Date: Monday, Dec.
</item 14>

<item 15>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
But the fresh grand dame of my tabletop world is Triangle Agency, a gleaming, razor-sharp new TTRPG that feels like a cross between The X-Files, Annihilation,
and playing Control with the Robert Anton Wilson mod turned on.

In Triangle Agency, you are super-abled agents investigating and stopping ontological anomalies from destroying reality, with mechanics that feel like the
platonic ideal mix of analytic crunch and grand narrative weirdness. And my god, it is gorgeous, with jaw-dropping art and design that puts it on the same shelf
as Wanderhome and Mörk Borg. The full game ships early in 2024, but you can get the digital rules and start playing right now, for the low, low price of
“whatever you think is appropriate.”

Twilight: 2000

Charlie Hall, senior editor, tabletop

I firmly believe that tabletop role-play should always take into account the audience that sits down at the table to play. That’s part of the reason why I hate
best-of lists for this particular category of games in particular.
</item 15>

<item 16>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Shohei Ohtani and Juan Soto have already found new homes, but much of the MLB offseason has yet to play out.

Japanese pitcher Yoshinobu Yamamoto is the focus of many teams as he prepares to sign a massive deal, while Blake Snell, Cody Bellinger and Jordan Montgomery
remain on the market. The trade market is also expected to be active once more dominoes fall, with Dylan Cease, Corbin Burnes and Tyler Glasnow among starters
who could be moved this winter.

The Winter Meetings are over, but because of the timeline of major decisions by Ohtani and Yamamoto, there could be a flurry of activity leading up to the end
of the year.

MORE: Why Shohei Ohtani signed with Dodgers over Blue Jays, other suitors

Sporting News is tracking the rumors, signings and trades as MLB free agency continues. Follow along below for the latest news.
</item 16>

<item 17>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
We have been promised very little of that, though. ESPN’s Jeff Passan reported this: “If visits between Ohtani and a team are reported publicly, it will be held
against the team.”

So, yeah. Probably won’t hear a ton, but that won’t stop the speculation. And it won’t stop us from passing along what precious few nuggets that are scattered
about.

MORE: Way-too-early MLB Power Rankings for 2024

Shohei Ohtani free agent rumors and updates

Ohtani has an 'affinity' for Boston, it seems

Date: Nov. 27

Source: Jeff Passan, ESPN

Hope everyone enjoyed the Thanksgiving holiday. Us baseball writers are pleased that Ohtani didn't decide to sign while the turkey was still warm. This isn't
exactly "breaking news" because rumors have been swirling, but it's an aside from an in-the-know reporter that just might give one fan base a little too much
hope and excitement.
</item 17>

<item 18>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Morosi reports Shohei Ohtani is "likely to decide on a team within the next week," according to a source.

Status quo on Yankees' Juan Soto trade talks

Date: Sunday, Dec. 3

Source: Jon Heyman, New York Post

The Yankees reportedly are pushing hard to acquire Juan Soto from the Padres, but Heyman says the teams haven't spoken since San Diego requested a six-player
package in return for Soto and Trent Grisham. That ask included pitchers Michael King and Drew Thorpe, who the Yankees aren't inclined to move. Stay tuned...

Shohei Ohtani field growing smaller

Date: Friday, Dec. 1

Source: Jeff Passan, ESPN

It has been expected since before he hit free agency that Ohtani would set the record for the largest fully guaranteed contract in North American sports
history, with Passan reporting the deal could reach as much as $600 million.
</item 18>

<item 19>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
At its best, you really can feel the Force around you. —CA

31. Connections

Developer: The New York Times

Where to play: Android, browser, and iOS

After the acquisition of viral hit Wordle, The New York Times’ Crossword ecosystem leveled up to full-fledged attention competitor — NYT is a media company with
a gaming platform. And its staff of puzzle writers and editors are not stooping to mind-numbing mobile content to keep the expansion going. Connections, its
latest title to pair well with morning coffee, is one of the year’s best games.

Connections offers you a grid of 16 words and a mission: Detect the common threads between four different sets of words without embarrassing yourself. The
grouping logic ranges from simple (“animals”) to silly (“synonyms for farting”) to sneaky (“countries when the letter ‘A’ is added”).
</item 19>

<item 20>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
He added there are believed to be five teams in the running for Ohtani.

Ohtani decision coming this week

Date: Dec. 3

Source: Jon Morosi, MLB Network

With the winter meetings set to begin in Nashville, it seems a resolution is near for Ohtani. MLB Network's Jon Morosi reports the superstar is "likely to
decide on a team within the next week," according to a source.

Ohtani race starting to clear up

Date: Dec. 1

Source: Jeff Passan, ESPN

Two days before the start of the winter meetings, the picture on Ohtani is beginning to clear a bit more. ESPN's Jeff Passan reported the Rangers, Red Sox and
Mets appear to be out of the running for Ohtani, while the Dodgers, Cubs, Blue Jays and Angels remain in the mix. The Giants are considered to be an unknown.
</item 20>

<item 21>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Shohei Ohtani, Yoshinobu Yamamoto decision not expected until after winter meetings

Date: Monday, Dec. 4

Source: Jon Heyman, The New York Post

The baseball markets are waiting for the first major domino to fall this offseason in free agency, but it does not appear as though that will happen before the
winter meetings conclude. Jon Heyman reports that neither Shohei Ohtani not Yoshinobu Yamamoto are expected to get a deal done before the final meetings on
Wednesday.

Wade Miley back with Brewers on 1-year deal

Date: Monday, Dec.
</item 21>

<item 22>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The Quiet Year

Em Friedman, associate professor of English at Auburn University and Polygon contributor

Look, The Quiet Year appeared not once but twice on last year’s roundup, I know. Jay Dragon praised its elegance and Keerthi Sridharan noted the way their game
group used it as a prologue (as Friends at the Table and The Adventure Zone have popularized), and it’s a standard early on in my TTPRG class. But for me it was
the standout game of this year because after running it for my niece (then 4) and nephew (8), my nephew asked that I run “the skulls game” for his entire class.
While not intended for a dozen grade schoolers, it turns out the mechanics of The Quiet Year — which ensure everyone gets to have their turn unimpeded by the
commentary of others, ritualizes dissent and discussion, as well as having a glorious sheet of blank paper to draw the zombie chicken monster friend of your
dreams — turned out to be a perfect framework to imagine how to work together.
</item 22>

<item 23>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: Bob Nightengale, USA Today; Ben Nicholson-Smith, Sportsnet

Ohtani had a change of plans, apparently. Hours after MLB Network's Jon Morosi reported that the two-way star was headed to Toronto for a meeting on Friday, USA
Today's Nightengale and Sportsnet's Nicholson-Smith reported that Ohtani is at his home in Southern California.

The Blue Jays are still finalists to land Ohtani but his landing spot is not official yet.

Ohtani headed to Toronto on Friday

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

Ohtani might not have officially picked a team, but he is headed to Toronto, per MLB Network's Jon Morosi. It had been previously reported he had picked the
Blue Jays, though several reports have since denied that he has made a decision.

Report: Ohtani picks Blue Jays over Dodgers

Date: Friday, Dec. 8

Source: J.P.
</item 23>

<item 24>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The whimsically illustrated rulebook lays out the tools by which you’ll lay bricks for your city, weaving together your ideas and your fellow players’ to
establish Landmarks, Residents, and Neighborhoods, all on a quest to find the city’s true names. It’s a beautifully simple game, guided by a Compass, which can
be anything — literally. It’s almost impossible for me to capture the magic of this game with words, because each session we played was so entirely unique, from
the verdant turtle-city in the sky we built out in stages to the haunted shell of a city we later used as our D&D setting. It’s a perfect game full stop, but
especially for those seeking to rekindle their creativity, or simply to sit down and create with friends.

Triangle Agency

“I laughed so hard my face hurt.”
</item 24>

<item 25>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The best games this year told their stories in interesting ways, be it ignoring your superpowers to bike through your old neighborhood in Queens, a musical
number, or fractured memories hidden in a puddle. But Baldur’s Gate 3 takes experimental storytelling to another level. Every choice and every new direction is
a simple success, fail, or crit away. One roll of the dice after another can take you all the way from a crashed Nautiloid to the city of Baldur’s Gate itself.

In a year filled with beautiful stories and powerful moments, it’s the spinning whir of Baldur’s Gate 3’s d20 that defined video game storytelling in 2023. —RG

1. The Legend of Zelda: Tears of the Kingdom

Developer: Nintendo EPD

Where to play: Nintendo Switch

Princess Zelda is missing. Ganondorf has returned, somehow, and he’s really hot. So an age-old hero in a green tunic has to step in.
</item 25>

<item 26>
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
</item 26>

<item 27>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
20

Source: Ken Rosenthal, The Athletic

Buried at the bottom of Rosenthal’s piece on what an Ohtani contract might look like — incentives, opt-outs and all sorts of stuff — is this little bit of
reporting:

An agreement at the winter meetings in Nashville the first week of December would be the ideal outcome for MLB, but not necessarily for Ohtani. He perhaps will
be better off waiting for the signing of Japanese righty Yoshinobu Yamamoto, who is expected to be posted for free agency as soon as Monday, triggering a 45-day
window for teams to negotiate with him. Losing out on the 25-year-old Yamamoto, who could command more than $200 million, might put certain teams on tilt,
improving the landscape for Ohtani.

Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov.
</item 27>

<item 28>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
But that would also be underselling the game, and understating things quite a lot.

Diving into the mysterious Blue Hole, Dave spends the first two quarters of his day swimming deeper into the colorful abyss, discovering both sea life and a
story that’s equally absurd and earnest. When you’re not picking up sea urchins or spearfishing sharks, Dave is assisting the rest of Dave the Diver’s cast of
characters — his sushi business partners, a community of seafolk, an anime-obsessed weapons expert, and a pair of dolphins. At night, Dave slings sushi and
pours drinks at the restaurant, frantically running back and forth between clearing dishes, delivering sushi, and refilling the freshly ground wasabi. Between
all that, Dave’s harvesting rice and vegetables on a farm, curating a hatchery, racing seahorses with mermaids, and taking down a suspicious group masquerading
as environmental activists. Somehow, there’s even a well-done rhythm video game — starring one of those anime idols that the arms dealer loves — that makes
perfect sense.
</item 28>

<item 29>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
I played a one-shot with some friends I was reconnecting with and we mostly group game mastered. We played a gang of messy screw-ups who didn’t hugely get
along, my character had traded her old gender to the fae over the summer, and we tried to investigate a potential monster hunter, a NPC most of our characters
found too cool to talk to. There was a gag about someone doing a club remix of “Danny Boy” at a memorial service and the group chat with those friends is still
named “Danny Boyz (The Pipes, The Pipes).”

Mörk Borg

Jasmine Bhullar, writer, content creator, executive producer of DesiQuest

Of the games I delved into this year, although several afforded hours of entertainment, I felt myself repeatedly returning to one in particular. Although Mörk
Borg was released in 2020, it was only in 2023 that I happily stumbled upon it and was able to somehow get a table together to run a game.
</item 29>

<item 30>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s very rare for a Soulslike to ever feel like anything more than a knockoff — even when they’re decent fun, like The Surge. But the best compliment I can
give Lies of P is that it feels like the genuine article, a FromSoftware game developed in an alternate dimension and somehow released in this one by mistake.
But it wasn’t a mistake or luck that made Lies of P, and it wasn’t FromSoftware, either; it was a talented group of developers at Neowiz Games and Round8 Studio
that took a tired genre, paired it with a bizarre IP, and knocked it out of the park. —Ryan Gilliam

Related Lies of P carves a singular space out of the Soulsborne genre

48. Tchia

Developer: Awaceb

Where to play: PlayStation 4, PlayStation 5, and Windows PC

Tchia is an open-world adventure game set in a fictional version of island nation New Caledonia — inspired by Awaceb’s co-founder’s childhood in the country.
</item 30>

<item 31>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Yoshinobu Yamamoto meets with Phillies

Date: Thursday, Dec. 14

Source: Jon Heyman, New York Post

The Phillies have not been one of the more frequent names connected to Yamamoto, however the Japanese free agent reportedly met with the team. The starting
pitcher has met with a number of teams, including the Dodgers, Mets, Yankees and Red Sox.

Philadelphia re-signed ace Aaron Nola this offseason, bringing back the powerful duo of Nola and Zack Wheeler. Adding Yamamoto would immediately give the
Phillies' one of the best rotations in baseball.

Cubs, Blue Jays the favorites for Cody Bellinger

Date: Wednesday, Dec. 13

Source: Jon Heyman, New York Post

The Blue Jays and Cubs were believed to be among the favorites to land Ohtani. But with the star two-way talent off to Los Angeles, the two are now believed to
be the favorites to land Cody Bellinger, according to Heyman.
</item 31>

<item 32>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
This year, we have the Avengers of rhythm games, Theatrhythm Final Bar Line, and Paranormasight: The Seven Mysteries of Honjo, an excellent riff on the visual
novel penned by a beloved storyteller — whose best series has never appeared in the U.S.

What should you know about Paranormasight before you play? Well, ideally nothing. Why else would I be eating up my word count?

But if you insist: It’s a mystery — and a horror mystery at that. You travel to 1980s Japan, specifically the Tokyo neighborhood of Honjo, located not so far
from the modern Tokyo Skytree. It’s hard to imagine that modern landmark ever towering alongside these streets, which are filled with shadows and lethal curses.

If you have even a passing interest in urban legends, spooky folklore, cults, and deadly rituals, or you’ve enjoyed series like Zero Escape and Danganronpa,
Paranormasight is an easy recommendation.
</item 32>

<item 33>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Playing Exquisite Biome feels a little bit like being David Attenborough on an alien planet.

With beautiful, evocative art and short, easy-to-parse rules, Exquisite Biome is my go-to “what if we made up some weird little guys and set them in motion for
an hour” game. And truly, what higher calling is there?

Fetch My Blade

Did you know you can play one of the best tabletop RPGs whenever you want, without any scheduling nightmares or unfeasible time commitments? You can when you
play Fetch My Blade, a solo journaling role-playing game created by Kelly Tran and Ethan Yen!

In Fetch My Blade, you role-play as a friendly or fierce dog of a retired legendary sword master. Create your character by choosing one of six dog breeds,
including shih tzu, corgi, mastiff, border collie, husky, and chihuahua, or select your own! Determine three truths about your master and aid them when a
mysterious stranger challenges them to a duel.
</item 33>

<item 34>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
What difference does it make if something scores highly on some arbitrary rubric if it doesn’t resonate with the people in your community? And so I look at
year-end lists like these as a kind of menu, boards of fare that clever game masters can use to pick just the right flavor for a given table.

For me this year was all about reconnecting with the other dads in my neighborhood after years spent locked inside during the pandemic. I spun up a Discord that
we used to organize friendly games of Call of Duty’s DMZ mode, organized a few visits to the local brewery to catch up make new connections, and generally just
tried to get awkward, middle-aged men accustomed to seeing each other socially again. One night I decided to bust out Twilight: 2000, and it was an instant hit.

Twilight: 2000 is a game about surviving in the aftermath of a fictional nuclear conflict in the heart of Europe. It’s an alternate history where the Cold War
went hot, swallowing up an entire United States Marine division.
</item 34>

<item 35>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I’ve rolled for every character, grinded out all of said characters, 100%-ed the exploration, completed every quest — did all the stuff that would burn anyone
out. Honkai: Star Rail is Hoyoverse’s answer to the Genshin Impact burnout.

The slower-paced turn-based combat, the ability to auto-battle (if your teams are strong enough), and the smaller maps make it the perfect daily game for me to
funnel my interest into without feeling exhausted. The characters are fun, flashy, and easy to latch onto. Should I main the sneaky but elegant Kafka, who uses
damage-over-time skills to whittle her opponents away? Or should I use the brooding Blade, who unleashes huge attacks at the cost of his HP? Ah, I guess I’ll
let my wallet decide — it is a gacha game, after all. —JL

20.
</item 35>

<item 36>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
After more than a decade of going on and on about the “board gaming renaissance,” it’s safe to say that tabletop role-play is finally having its moment in the
spotlight. But while Dungeons & Dragons’ OGL fiasco has clearly lit a fire under many players eager to make a change, the fact of the matter is that a sizeable
cohort of writers and designers have been toiling away at excellent games for years now. Their tireless efforts have yielded a bumper crop of excellent, some
might say genre-defining, TTRPGs. Just like board games, a few modern classics have also cropped up: Look no further than The Quiet Year and The One Ring, both
making their second appearance on this list.

Just as before, Polygon asked nearly two dozen writers, designers, presenters, actors, and personalities from around the world of tabletop gaming to share with
our readers the TTRPGs that made an impact on them and their players this year. Here’s what we found.
</item 36>

<item 37>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
Dogmatic choices exult the God-Emperor and loathe the mutant and the witch, the Heretical options pursue corruptive power, and the Ionoclast path is the closest
thing we have to modern-day “good guy” morality.

After the events of the tutorial, my beautiful voidship is run down and my staff is struggling. As the newly anointed Rogue Trader, I have to trek around the
various planets of the Koronus Expanse to get a new Navigator, fix my ship up, and avoid any major diplomatic incidents with the locals. My voidship is the size
of a modern city; I’m as much a governor as a captain, and I have to manage the ship, its cargo, and its many occupants.

In the process of getting back on my feet, I uncover a nefarious cult and a deep conspiracy. It’s not an easy job, but I’m blessed with a handful of companions
from the Imperium to help me out.
</item 37>

<item 38>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
That’s gotta be super disappointing for Mariners fans starved for a title, right?

On the other hand, maybe the M’s know all about Ohtani’s desire to keep the rumor mill quiet, and leaking “news” like this would please Ohtani and his camp.

‘Timing could be right for a Cubs-Ohtani union’

Date: Nov. 16

Source: Jesse Rogers, ESPN

The Cubs were one of the seven finalists for Ohtani the first time around, but lack of the DH in the NL at the time made it hard for Ohtani to choose Chicago.
His primary goal was showing he could both hit and pitch at a high level for an MLB team, and trying to play the outfield instead of just DHing would have been
too much, even for Ohtani.

Now, though? The NL has the DH, and the Cubs are itching to make a sport-shaking signing. Well, another sport-shaking signing.
</item 38>

<item 39>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
The victory tasted like ash in my mouth after all that frustration.

Voidship combat is another aspect that feels clunky and frustrating. Like the squad-based skirmishes, naval encounters are also turn-based, where positioning is
ultra important. Space naval battles should feel tense, but instead, I’m mostly annoyed at having to continually rotate my ship and set up my zones of attack. I
wish I could delegate these annoyances to my Senechal — to delegate the duties of character leveling and ship combat, the better to appreciate all of the things
Rogue Trader is doing so well.

Rogue Trader is a dense, vast game, and much of it has clearly been crafted with love for the expansive lore of the 40K canon. While there are small annoyances
and clunky features along the way, the political intrigue, cast of characters, and moral choices have me hooked.
</item 39>

<item 40>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
While he won't pitch again until 2025, Ohtani's power and on-base ability gives Los Angeles the most dangerous trio of hitters in the sport.

MORE: How Dodgers' 2024 payroll looks after handing out monster deal to Shohei Ohtani

Red Sox swing deal for Cardinals' Tyler O'Neill

Date: Friday, Dec. 8

Source: Jon Heyman, New York Post

Days after trading Alex Verdugo to the Yankees, the Red Sox have acquired OF Tyler O'Neill from the Cardinals in a deal that sends pitching prospects Nick
Robertson and Victor Santos to St. Louis.

O'Neill enjoyed a breakthrough 2021 season with the Cardinals, batting .286 with 34 home runs and a .912 OPS, but he's struggled to recapture that level of
production over the last two seasons and is set to be a free agent after 2024.
</item 40>

<item 41>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 41>

<item 42>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And there is already loads of content to support that kind of time investment. —RG

Related Diablo 4 activates and frustrates my lizard brain

27. Hitman World of Assassination: Freelancer mode

Developer: IO Interactive

Where to play: PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

Hitman World of Assassination’s Freelancer mode, which debuted in January and puts a roguelike twist on Agent 47’s globe-trotting murder-for-hire missions,
isn’t for the faint of heart. It demands constant improvisation, to a degree that can challenge even veteran Hitman players. Repeated failures can make it feel
more frustrating than fun.

But as every fan of roguelikes knows, these high-stakes experiences have the capacity to deliver a sense of exhilaration like nothing else. Pulling off multiple
daring kills, hiding the evidence (or going out with a bang), and trying to make it out alive with all the gear you brought into the mission — it’s tense and
thrilling, every time.
</item 42>

<item 43>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
“In the grim darkness of the far future, there is only war.” That’s the tagline of Warhammer 40,000, one of the most over-the-top and brutal sci-fi settings
around. But even in a merciless, brutal dystopia that grinds its people into dust, some get to enjoy being on the top of the food chain. Warhammer 40,000: Rogue
Trader is about the perils and pleasures of being atop that particular hierarchy.

Rogue Traders in 40K are freelance explorers, scouting the far frontiers of the Imperium of Man. It’s a risky career, but it comes with luxurious rewards:
agency, freedom, power, and a giant flagship. As the player, I explore a system of the Imperium with a loyal crew of companions, making impactful choices and
determining the fate of those around me.

Rogue Trader is a computer role-playing game in the vein of Baldur’s Gate 3 or Pillars of Eternity.
</item 43>

<item 44>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
The Hot Stove dominoes have started to fall, but everyone who loves baseball — fans, players, coaches, front-office types and everyone else — is waiting for
free agent Shohei Ohtani’s decision. Where will the global superstar sign his next contract?

He’s played his entire six-year MLB career with the Angels, but his teams haven’t even sniffed the postseason and Ohtani’s made his desire to win crystal clear,
so he won’t be back. Will he chase every dollar possible, going solely by the highest offer? That seems unlikely, too. Remember, Ohtani wants to win.

Lots of teams can meet Ohtani in that middle ground, offering both lots of money — he will absolutely sign the largest contract in baseball history — and lots
of chances to compete for a World Series title (or two or three). It’s a long list.

MORE: Ranking 15 possible destinations for Shoehei Ohtani

So, the baseball world waits for news.
</item 44>

<item 45>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Here's the latest on Ohtani's free agency pursuits.

MORE: Projecting Shohei Ohtani's contract in free agency

Shohei Ohtani news: Latest updates, rumors for MLB free agency

Ohtani agrees to 10-year, $700 million deal with Dodgers

Date: Saturday, Dec. 9

Shohei Ohtani announced Saturday that he has agreed to a contract with the Dodgers. It's a colossal 10-year, $700 million deal with "significant deferrals," so
while Los Angeles won't pay him $70 million per year, the franchise will be paying him beyond those 10 seasons.

While rumors connected Ohtani to the Blue Jays on Friday, it's the Dodgers who hand him a record-shattering contract and pair the two-time MVP with two former
MVPs in Mookie Betts and Freddie Freeman.

Ohtani not headed to Toronto, at his home in Southern California

Date: Friday, Dec.
</item 45>

<item 46>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
4

Source: Ken Rosenthal, The Athletic

While one report earlier Monday indicated Ohtani isn't expected to sign before the end of the Winter Meetings, the process seems to be moving along.

The Athletic's Ken Rosenthal reports Ohtani is "believed" to have met with the Blue Jays at the team's complex in Florida on Monday, far away from the meetings
in Nashville. The Blue Jays have been pursuing Ohtani and Juan Soto in hopes of striking a deal for one of the two stars.

Braves are reportedly 'engaged' with Ohtani's camp

Date: Dec. 4

Source: Jon Morosi, MLB Network

It's not surprising that any and all teams should be interested in Ohtani's services, but imagine the Braves landing the superstar? According to MLB Network's
Jon Morosi, the Braves are "engaged" in talks with Ohtani's camp.
</item 46>

<item 47>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
All eyes since the start of the offseason have been on Shohei Ohtani. And for the most part, there hasn't been much to see.

The two-way phenom is the most sought-after free agent in MLB history, with his combination of talent and marketability making him a franchise-changing player.

And the bidding for Ohtani could be winding down soon. Multiple reports have indicated Ohtani could sign at some point during the winter meetings, which are
scheduled to take place from Dec. 3 through Dec. 6.

Many of the top teams are expected to be in the mix for Ohtani. The Dodgers and Cubs are among the top teams expected to be aiming to sign the two-time AL MVP,
and plenty of others should continue to compete through the end to land him.

Here's the latest on Ohtani's free agency pursuits.
</item 47>

<item 48>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It’s a time in your life where you feel so, so much. You can see that earnestness in the teenage experience where everything is a big, huge issue — sometimes to
the point of cringe — tied up in that big, global issue of the meteor that’s looking to destroy everything. —N. Carpenter

Related August games you might have missed

40. World of Warcraft Classic

Developer: Blizzard Entertainment

Where to play: Mac and Windows PC

Something very, very interesting started to happen in the retro version of WoW this year. While Blizzard has been content to march one half of the game’s
community forward through its history of expansions, it’s started to come up with creative ways to keep the other half — the half that wants to stay in the game
as it was at launch — engaged.

The first of these was Hardcore, a brilliant permadeath mode that instantly made this aging game both more dangerous and more social, resurrecting the spirit of
its 2004 servers.
</item 48>

<item 49>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
'Money's great, but winning's better'

Date: Nov. 21

Source: Max Scherzer, Rangers pitcher on Foul Territory

OK, so this isn't necessarily a rumor, but an example of how much players would love for Ohtani to join their team. Scherzer was a guest on Foul Territory on
Monday.

"Money's great, but winning's better."



Max Scherzer shares his pitch for Shohei Ohtani to join the @Rangers 🏆



▶️ https://t.co/6KoSRqSgk2 pic.twitter.com/PHW2jHWj6C — Foul Territory (@FoulTerritoryTV) November 20, 2023

Maybe Ohtani won’t sign quite so soon after all

Date: Nov.
</item 49>

<item 50>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Juan Soto to Yankees 'close to being finalized'

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

A trade between the Yankees and Padres for Soto is reportedly "close to being finalized," with right handers Michael King and Drew Thorpe expected to highlight
the return to San Diego. MLB Network's Jon Morosi reported the package will include at least two more players, neither of which will be Alex Verdugo.

Ohtani decision expected to be made before end of weekend

Date: Wednesday, Dec. 6

Source: Jon Morosi, MLB Network

MLB free agency remains largely on hold as teams wait for Ohtani to sign, and though it had at at point been reported he could sign during the winter meetings,
there appears to be a new timeline. MLB Network's Jon Morosi reported Ohtani is expected to make his pick before the end of the upcoming weekend. The winter
meetings end on Thursday.
</item 50>

<item 51>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
For 40K fans, this is a rare treat — a game that digs past the heroic facade of bolters and battles and taps into the grimdark dystopia that makes this
particular sci-fi setting so damn compelling.

Warhammer 40,000: Rogue Trader will be released on Dec. 7 on Windows PC. The game was reviewed using a pre-release download code provided by Owlcat Games. Vox
Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via affiliate links.
You can find additional information about Polygon’s ethics policy here .
</item 51>

<item 52>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The second, just launched, is the wild Season of Discovery, which remixes and restructures the original WoW experience — interpolating staggered level caps,
shuffling class roles — in ways that might just change MMO design forever. WoW Classic is quietly, and paradoxically, where Blizzard is doing its most
forward-thinking work right now. —Oli Welsh

39. Fire Emblem Engage

Developer: Intelligent Systems

Where to play: Nintendo Switch

Fire Emblem Engage was designed for a very specific kind of sicko: one not particularly interested in the origin stories of a horde of teenagers, or the
politics of a bourgeoise academy, or what kind of tea a teacher prefers, but instead one obsessed with the endless minutiae of combat stats, weapon loadouts,
and team composition. I know this because I am one such sicko.

If you’ve read any of my reviews or essays on Polygon, then you know I prefer strategy games that can get out of their own way.
</item 52>

<item 53>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Mike Mahardy

How the Polygon top 50 list works

Over the past few weeks, the Polygon staff voted, championed, debated, and ultimately threw up its hands and marveled at the list of mammoths, curiosities,
puzzle boxes, and black holes that is our top 50 games of 2023. Any video games that were released in 2023, received substantial updates in 2023, or achieved
renewed cultural relevance in 2023 were eligible for this list. Last year, the cutoff for consideration was Nov. 30. (You’ll notice a certain Firaxis Games
joint fairly high up our list.) This year, the cutoff was the same. Should we be thoroughly enamored with Warhammer 40,000: Rogue Trader or Avatar: Frontiers of
Pandora, we’ll make sure they’re considered for next year’s top 50.

Top 50

50. Mr. Sun’s Hatbox

Developer: Kenny Sun

Where to play: Nintendo Switch and Windows PC

Mr.
</item 53>

<item 54>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Much like humanity itself? —ML

23. Paranormasight: The Seven Mysteries of Honjo

Developer: Square Enix

Where to play: Android, iOS, Nintendo Switch, and Windows PC

Square Enix has plenty of mega-franchises to fill its time (and its coffers). This year, we have new entries for Octopath Traveler and Final Fantasy, along with
new Dragon Quest and Kingdom Hearts games in the not-so-distant future. Dayenu!

And yet, the publisher can’t help itself from bombarding us with surprising, interesting, sometimes great, often good-enough experiments. In 2022, we got an
English-language remake of lost gem Live A Live, the surprisingly enjoyable tactical RPG DioField Chronicle, a bonkers Final Fantasy spinoff featuring the
musical stylings of Limp Bizkit, and a pair of oddball card games lathered in lore from gaming’s best weirdo.
</item 54>

<item 55>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Those two would be the biggest supporting bats in the lineup around Ohtani. However, both are free agents after the 2025 season, and given the added payroll of
signing Ohtani, it's possible Toronto would view keeping the other two phenoms as too pricey.
</item 55>

<item 56>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Dodgers officials declined comment when asked if Ohtani had taken a physical, the final step before a deal can become official.

Significant questions linger about Ohtani’s future. He will not pitch in 2024 as he recovers from a September operation to repair his right elbow’s ulnar
collateral ligament. He underwent Tommy John surgery in 2018. Neither Ohtani, who has not taken questions from reporters since August, nor his agent, Nez
Balelo, nor the Angels have disclosed the exact nature of the second surgery, but the Los Angeles Times has reported it was a second Tommy John procedure.

Balelo has stressed that Ohtani remains committed to both pitching and hitting in the future. “Shohei loves to pitch,” Balelo told reporters in September.
Ohtani will attempt to return to the mound in 2025. His camp has not revealed at what point Ohtani would consider giving up his dual career and focusing on
learning a different position.
</item 56>

<item 57>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
You play as a broody Clive Rosfield, a young man whose life’s work is to protect his little brother, Joshua. The story begins when Clive’s life takes a turn for
the worse and he vows to destroy the monster who ruined his and his family’s legacy.

Developed by Creative Business Unit III, Square Enix’s internal team behind the MMORPG Final Fantasy 14, 16 leans into patchwork territories of fantasy genre
fare. There is palace intrigue, a whole lot of sex, and endless war between nations. But the developers then sprinkle in Final Fantasy elements like mother
crystals, dazzling kaiju fights between summons (known as Eikons in this iteration), and of course, Chocobos.

The quality of the story in this long and linear character-driven RPG waxes and wanes, but the action combat is among the best I’ve ever played. The gameplay
grips you from the very beginning as Clive smoothly dashes, parries, and swings his giant sword and varied magic with a dazzling amount of style.
</item 57>

<item 58>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Triangle Agency

“I laughed so hard my face hurt.” —me, 10 minutes after my first session of Triangle Agency

I love a game with style and a clear, fun point of view, so I’ve been a fan of Triangle Agency since I picked up its player guide in beta form earlier this year
(to prep for writing a published adventure for the game). Written in the form of a cheerful, slightly threatening employee handbook, the Field Agent Manual
guides you through the creation of your supernaturally powered character, then gets you ready for your role finding and containing Anomalies that threaten the
nature of reality. After your team’s morning meeting, of course.

Beyond the style of the materials, though, playing the game was an absolute blast. The mix of mundane life, weird occurrences, and corporate BS created a
fast-paced game experience that was playful and easy to understand (just count those threes!
</item 58>

<item 59>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
As far as games go, there are probably very few out there that require such a strict fidelity to the road laid out before you, but City of Winter uses this
structure to facilitate the kind of deeply resonant character arcs and development that many role-players crave in their games. An astoundingly complex game,
bringing your best stories to the table will leave you with a sense of loss and belonging that seem to walk together, hand in hand.

Dead After Dinner

Jason Morningstar, owner, creative director, and lead designer at Bully Pulpit Games

Jenn Martin’s game Dead After Dinner combines the relentless procedural revelation of For the Queen with Knives Out-style murder mystery, and it is endlessly
hilarious. Everyone is part of a miserable family stuffed with resentment, somebody is getting killed, there will be uncomfortable questions, and a murderer
will definitely be revealed. Or will they? Dead After Dinner is a truly pick-up-and-play murder mystery that will delight you and your scenery-chewing friends.
</item 59>

<item 60>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It feels forbidden somehow, even as it’s obviously harmless.

But Dixon also captures something ephemeral and nostalgic about what it’s like to be in a niche online community and group chat, and the magic of making
discoveries in those spaces. On an increasingly centralized internet, that experience has become rare, and I miss it.

Exquisite Biome

Everest Pipkin, game and software developer

I’ve heard Exquisite Biome described as “Spore for the kitchen table,” and if that isn’t a great pitch I don’t know what is.

Coming from Caro Asercion (game designer) and Si Sweetman (illustrator), Exquisite Biome is in the world-building TTRPG tradition, but it focuses specifically
on ecological biomes and the creatures that populate them. Through short generative prompts tied to a deck of playing cards, you build and witness the lives of
animals far beyond the edge of our own world. Playing Exquisite Biome feels a little bit like being David Attenborough on an alien planet.
</item 60>

<item 61>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
—Julia Lee

Related September games you might have missed

45. The Talos Principle 2

Developer: Croteam

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like its predecessor, The Talos Principle 2 tackles grand science fiction ideas, particularly about what it means to be human — a theme thoroughly explored in
this installment of the series set in a post-human society of AI-powered robots that are carrying on human culture and civilization. Also like its predecessor,
The Talos Principle 2 is replete with philosophical commentary and references to famous artists and thinkers. My favorite is a riff on Werner Herzog’s famous
quote about birds: “The enormity of their flat brain. The enormity of their stupidity is just overwhelming.”

But if you’re like me, you’re playing the game because you’re an absolute freak for light refraction, gravity, and geospatial puzzles.
</item 61>

<item 62>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I love the characters, the environments, the writing, the lore, and the flow of battles. But I have concerns with the game’s pacing. By the end of the first
chapter, I had leveled up 16 times. Each level offered marginal rewards, like being able to move slightly farther during the character’s turn in combat or
having a higher parry chance when being attacked. A slow drip-feed system means each level feels less important, and even though I’m growing stronger, I don’t
get that sense of long-term satisfaction.

It’s especially frustrating to hit a roadblock like the one at the end of Act 1, where an incredibly tough boss rolls out of nowhere and spanks my crew — and I
can’t leave to go grind experience somewhere else. I eventually found out a way to cheese the fight by focusing on my melee fighters’ positioning, but it took
far too long banging my head against the wall. The victory tasted like ash in my mouth after all that frustration.
</item 62>

<item 63>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Furthermore, in the process of putting a game together, I found a glut of free online tools from dungeon generators to NPC character sheet generators that made
game prep a breeze. I find myself coming back to this game again and again with each game session only giving me ideas for future campaigns. I know I’m not
alone in this feeling given the number of Mörk Borg-compatible games that have sprung up this year, from Pirate Borg to Chris Lockey’s upcoming Abyss of
Hallucinations.

I’m Sorry Did You Say Street Magic

Madison Durham, writer and journalist, staff writer at Reviewed

To me, some of the most joyful moments in tabletop gaming come from the experience of creative collaboration — the magic of making something new with friends,
be it a narrative, a character, a world, or in this case, a city. I’m Sorry Did You Say Street Magic by Caro Asercion is a GM-less city-building game, played
between a group of two to six people with a stack of index cards and your favorite writing utensils.
</item 63>

<item 64>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
He still has a lot left to give to the diamond for approximately 6 more years. If the Dodgers have the chance to draft them, should they go with Ohtani, Trout,
both, or neither? Let us know your opinions in the comment box.

Watch This Story: Shohei Ohtani’s Two-Way Dream: Before Injury, He Always Wanted Both!
</item 64>

<item 65>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In much the same way, the gameplay feels surprisingly modern at a distance, but on closer inspection, you start to see how this proto-immersive sim is actually
what inspired so much modern game design. You have to rely on your own curiosity, caution, and cunning to navigate the halls of Citadel Station and upgrade your
hacker into a cybernetic death machine. It’s a game that expects a lot from the player (and a little save scumming), but the experience is just as rewarding as
it was in 1994. —CA

Related The System Shock remake does something remarkable

32. Star Wars Jedi: Survivor

Developer: Respawn Entertainment

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Is Star Wars Jedi: Survivor an Empire Strikes Back level of sequel? Well, no, but it manages to get extraordinarily close to being one of the best follow-ups in
the entire Star Wars franchise.
</item 65>

<item 66>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Well, another sport-shaking signing. They already poached Craig Counsell from the rival Brewers to be their new manager, and it would be shocking if they didn’t
make a corresponding player-acquisition move. Ohtani’s at the top of the list.

Ohtani could potentially sign before Winter Meetings

Date: Nov. 11

Source: Alden Gonzalez, ESPN

Pretty much everyone in baseball would LOVE for this to happen. The team that winds up signing him, of course, will kick the Ohtani hype machine into gear ASAP.
Not only is there merchandise to sell, but the front office will use that signing to help lure other potential free agents, maybe even for a “play with Ohtani”
discount. But even for the other teams, knowing that Ohtani is off the board relatively soon into the offseason would allow them to shift to other priorities,
like trading for Juan Soto or deciding which free-agent pitchers to pursue with the money that might have been earmarked for Ohtani.
</item 66>

<item 67>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I played Marvel’s Avengers, too, and this isn’t that. It might seem like it’s going to be at first, because Midnight Suns makes the grave error of introducing
Iron Man and Doctor Strange as its tutorial characters, and these two might just be the most irritating characters in the entire video game. (I have beaten the
game, so I am allowed to make this call.) You must press on and give Midnight Suns time to win you over. Because it has so, so much more to offer than it may
appear in its first few hours.

Picture the romance and humor of Fire Emblem: Three Houses, combined with the high-stakes tactical battles of XCOM 2 — that’s what Midnight Suns becomes in its
mid-game and endgame. It’s a card-based strategy game, and each hero has their own customizable deck. I started off favoring Captain Marvel, Magik, and Blade,
simply because their moves and hilarious dialogue kept me entertained, but I soon realized that every single character has something exciting or unexpected to
bring to the battlefront.
</item 67>

<item 68>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
He’s cruel, depraved, and an absolute hoot. Yes, he may flay a few too many people for my tastes, but he’s the best companion to bring to a party.

In the grand scheme of things, this is one of the most complete and detailed explorations of the 40K universe you can find. The game is an homage to the
Warhammer 40K RPGs from Fantasy Flight Games, including Rogue Trader and Dark Heresy. I spent years as a teenager and young adult poring over these sourcebooks
(which Rogue Trader is inspired by), learning more about the tiny details of life in the Imperium. Developer Owlcat Games has paid the same attention to every
detail of the Koronus Expanse. I delve into ancient facilities staffed by tech-priests of the Machine God, the hostile xenos city of Commorragh, or massive
cities built to honor the God-Emperor of Mankind.
</item 68>

<item 69>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
He did this while making starts and dealing with a torn ligament in his elbow.

Only one stage remains for Ohtani. He has never played a postseason game in the majors. As the Angels foundered in recent years, Ohtani became more vocal about
his desire to play for a winner. Now he has the chance to make a different kind of history.

Required reading

(Photo: Stacy Revere / Getty Images)
</item 69>

<item 70>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
8

Source: J.P. Hoornstra, Dodgers Nation

The Blue Jays have reportedly won the sweepstakes for Ohtani. According to Dodgers Nation's J.P. Hoornstra, Ohtani has selected the Blue Jays. The official
details have not been reported, though the report indicated it will shatter Mike Trout's record 12-year, $426.5 million deal.

However, Sportsnet's Ben Nicholson-Smith reported Ohtani has not yet made his decision, and there is not a timeline, though he noted Toronto is a finalist.

Decision from Ohtani 'imminent'

Date: Friday, Dec. 8

Source: Jon Morosi, MLB Network

The wait could soon be over. MLB Network's Jon Morosi reported Ohtani is expected to make his decision soon, reporting it could come as early as Friday.
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec.
</item 70>

<item 71>
MLB free agency tracker: Live updates on news, rumors, signings and trades after 2023 Winter Meetings (Sporting News, 2023-12-23T23:54:00+00:00)
Pavlovic added that it is very likely that Yamamoto ends up either in Los Angeles with the Dodgers, or New York with the Mets or Yankees.

Yamamoto met with Giants, won't decide on team until after Christmas

Date: Tuesday, Dec. 19

Source: Andrew Baggarly, The Athletic and Bob Klapisch, Newark Star-Ledger

San Francisco has been among the teams most heavily believed to be in on Yamamoto, and according to The Athletic's Andrew Baggarly, the team met with him in San
Francisco last week. Baggarly reported the Giants have been willing to "blow teams out of the water" to land Yamamoto, but noted he believes the crowded picture
for the Japanese ace could make it more challenging for the Giants to be the winner of the Yamamoto sweepstakes.

And fans will have to wait a bit longer before hearing a final resolution on his decision.
</item 71>

<item 72>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In Alan Wake 2, everyone’s story is happening at once, branching out in endless directions, and we might not know the genre until it’s too late. It’s a
hypertext mystery, an ergodic game that molds itself to an audience raised online, though few computers are in sight. It’s a game for a world suspended in an
endless second act — forever wars, live services, franchises, infinite scrolls — endlessly searching for a conclusion. —JR

2. Baldur’s Gate 3

Developer: Larian Studios

Where to play: PlayStation 5 and Windows PC

Baldur’s Gate 3 hit at the perfect moment. Despite the handful of truly great computer role-playing games that have been released since the early 2000s (Larian
Studios itself made waves with its Divinity: Original Sin series in 2014 and 2017), the genre has largely remained a modern niche. But times have changed:
Dungeons & Dragons has undergone something of a renaissance, thanks to actual-play series like Critical Role and Dimension 20.
</item 72>

<item 73>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It uses custom tarot cards and a whiteboard to track your character’s progress as they dare to shine a light on the chilling danger of a variety of potential
scenarios full of ghosts and other dark entities.

​The action is propelled through the use of the custom tarot cards — the Night Deck — which are filled with prompts that require the creative interaction of
players. Each round, players take turn as the director of the story and guide the narrative forward using the guidance of the cards to describe and draw each
room of the haunted house that they are exploring. The game encourages collaboration in determining the story’s direction and does a remarkable job of steering
the action forward without a single game master putting forth the often large amount of preparation that role-playing games usually require.

​As the game progresses, characters collect wounds and push past the terrors that await them and work together to resolve the mysteries at the root of each
horror. The solo mode of the game plays just as smoothly as the group experience, leaning on the prompts and player’s imagination to create a journal
documenting the character’s harrowing encounter.
</item 73>

<item 74>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Like all great RPG supplements, The Monster Overhaul not only offers answers for these questions and more, it also teaches the reader how to continue answering
them long after these published tables and suggestions are exhausted. A monstrous achievement that should be on every GM’s shelf.

Bonus: Reach of the Roach God

Reach of the Roach God is the hardcover high-watermark of A Thousand Thousand Islands (ATTI), a series of zines, that presents a system-agnostic setting
inspired by the folklore of Southeast Asia. Odoyoq, the roach god, senses new ways it can insinuate itself into the lives of non-roachkind. The three schemes
are presented as both traditional adventures and as collections of facts — about the setting, about the people living there and their motivations, about the
desires of the god and the machinations of its agents. Events will quickly throw a small village, a monastery, and a necropolis into chaos. Two gazetteers and a
sourcebook on the roaches follow, giving the GM ample material to expand the campaign.
</item 74>

<item 75>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
It appears that might no longer be the case. USA Today's Bob Nightengale reported one high-ranking executive believes the Cubs' optimism of signing Ohtani "has
now significantly wanted." The report stated that leaves the Dodgers, Blue Jays, Angels and Giants as the remaining finalists.

Ohtani has 'positive feeling' about Blue Jays

Date: Tuesday, Dec. 5

Source: Jon Heyman, The New York Post

The Blue Jays reported met with Ohtani on Monday, but even before he went to talk with the team, Ohtani reportedly had a "positive feeling about Toronto," along
with other cities and teams, including the Dodgers. He has been linked to Toronto and Los Angeles throughout the process.

Blue Jays 'believed' to have met with Shohei Ohtani

Date: Monday, Dec.
</item 75>

<item 76>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
The Giants are considered to be an unknown.

In addition to the teams, Passan also reported Ohtani is expecting a contract somewhere between $550 and $600 million, which would be the largest guaranteed
contract in North American sports.

Blue Jays could alter roster with Ohtani signing

Date: Nov. 30

Source: Jon Heyman, New York Post

The Blue Jays are one of the teams believed to still be among the strongest in the mix for Ohtani. The New York Post's Jon Heyman reported there could be some
changes if Ohtani signs with the Blue Jays, however, reporting that some rivals believe Bo Bichette and Vladimir Guerrero Jr. could hit the trade block if
Toronto lands the star.

Trading the two infielders might seem counterproductive to signing Ohtani in the first place. Those two would be the biggest supporting bats in the lineup
around Ohtani.
</item 76>

<item 77>
M.L.B. Superstar Shohei Ohtani to Sign $700 Million Deal With Dodgers (The New York Times, 2023-12-09T22:04:02+00:00)
Since he was a teenager, Ohtani has ignored suggestions that he focus on only one pursuit.

Ohtani demonstrated his potential as the American League Rookie of the Year in 2018, but his two-way hopes were delayed after his first elbow surgery. It was
not until 2021 that the full flower of his ability bloomed. He has won the American League MVP in two of the past three seasons; in the intervening season, he
led all American League pitchers in strikeout rate while hitting 34 home runs with an .875 OPS. To create a comparison for him involves inventions that sound
freakish. “It’s like if Judge went out and was a 20-game winner as well,” former teammate Kole Calhoun said.

Ohtani will not pitch in 2024 after undergoing Tommy John surgery, but his agent stressed he’s committed to returning to the mound. (Photo: Michael Owens /
Getty Images)

Ohtani is committed to being a starting pitcher.
</item 77>

<item 78>
Unraveling Potential Reasons Why the Dodgers May Not Pursue Shohei Ohtani and Mike Trout in Free Agency Despite Their Availability (Essentially Sports,
2023-11-18T21:21:03+00:00)
The Los Angeles Angels have gone through a tough year. After a heartbreaking 2023 season, they now have to cope with the impending exit of their star two-way
star, Shohei Ohtani. The team is also open to trading their nine-time Silver Slugger hitter Mike Trout, should he request it. In a huge miss, a team that
couldn’t do well even with two of the league’s biggest talents on their roster may lose said stars at the same time. Will they choose to stay in LA?

America’s Favorite Video Today

The Halos’ cross-town rivals, the Los Angeles Dodgers, have been eyeing the dynamic Troutani duo for a long time. However, despite fans’ expectations, the
Dodgers’ Ohtani-Trout conquest might not take place—and for good reasons.

Do the Dodgers have the budget to afford Ohtani and Trout on the same roster?
</item 78>

<item 79>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Interesting, certainly.

Mariners out of the Ohtani chase?

Date: Nov. 17

Source: Daniel Kramer, MLB.com

Here’s the key piece of the article: “Industry sources told MLB.com this week that landing Ohtani doesn’t appear to be within the Mariners’ realistic agenda
this offseason.”

In a way, that’s surprising. The Mariners seem to check a lot of boxes for Ohtani. They’re set up to win for a long time with a bunch of young starting pitchers
and a co-superstar in Julio Rodriguez. Plus, Ohtani would get to be part of the quest for the franchise’s first World Series title, he’d get to wear the same
uniform as his baseball hero, Ichiro, and he’s lived there in the offseason. So it must be about the money, and Seattle’s reluctance to even get in the
conversation. That’s gotta be super disappointing for Mariners fans starved for a title, right?
</item 79>

<item 80>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
For the first year in recent memory, scale did not necessitate tradition, and scope did not preclude getting weird. In 2023, nothing was sacred in video games,
and so they felt more vibrant than ever.

Sure, some of the more “focused” games threw us for a pleasant loop: Dredge begins as a lonely fishing sim before transforming into something otherworldly, and
Humanity morphed from a pensive art project into an all-out war. Dave the Diver, similarly, is not so much about being a diver as it is about running a sushi
restaurant, or hunting for alien artifacts, or conversing with said aliens, or — you get the point. Whether you booted up your Steam Deck for a cross-country
flight or hid your Switch off screen during that boring Zoom meeting, the game you returned to was rarely the one you left behind.

This amorphousness (I’m begging our copy editor to let this one slide, because what other “word” could adequately summarize the video games of 2023?)
</item 80>

<item 81>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
wasn’t consigned to the newcomers, though. Larian Studios, fresh off two years of early-access development and riding the reputation it had garnered from
Divinity: Original Sin 2, saw fit to release a role-playing game in which you can kill off nearly every main character the moment you meet them. Remedy
Entertainment — let’s be honest, this group has always been strange — made a sequel that’s equal parts horrifying, hilarious, fun, and fabulous. And Nintendo?
Well, Nintendo had another banner year. That’s no surprise. The real surprise? It finally let go, and let players toy with the digital molecules of its most
revered series. More on this below.

As the year comes to a close, it’s intoxicating to see developers of all sizes, in every genre, with every tier of budget, mining the depths of interactive
design, branching this way and that as they follow their respective veins of gold. They’re nowhere near the bottom of that particular expanse, of course — and
that’s a heartening thought.
</item 81>

<item 82>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Coincidentally, Ohtani made his original decision to sign with the Angels on Dec. 8, 2017, six years to the day.

Dave Roberts says Dodgers met with Ohtani

Date: Tuesday, Dec. 5

Source: Alden Gonzalez, ESPN

Dodgers manager Dave Roberts told reporters Tuesday that the team met with Ohtani for 2-3 hours over the weekend. Roberts said Ohtani is "clearly" the Dodgers'
top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a timeline for Ohtani's decision.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec. 5

Source: Bob Nightengale, USA Today

The Cubs have been viewed as one of Ohtani's likeliest destinations for a while. It appears that might no longer be the case.
</item 82>

<item 83>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
More precisely, I love when strategy developers can put their pens down, throw their hands up, and admit that the stories unfolding in the player’s head will
almost always be more powerful than anything they could write. Fire Emblem Engage is one of the foremost proponents of this idea. It hurls an excess of
characters, weapons, battle scenarios, and stat-boosting abilities at you, leaving the door open for you to observe character interactions on the battlefield
and create the resulting fanfiction in your head. Its actual script is a quagmire of nonsensical JRPG tropes, and each cutscene is more skippable than the next.
But if you’re looking for an excellent turn-based tactics game that gets out of the player’s way, you can do a whole lot worse than Fire Emblem Engage. —M.
Mahardy

38. Pizza Tower

Developer: Tour De Pizza

Where to play: Windows PC

Pizza Tower is a perfect object, and fully committed to its vision.
</item 83>

<item 84>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
), got every player invested in our characters and each other (even in a group of total strangers), and allowed the whole table to lean into both the strange
and ridiculous nature of life. This year, it was just the kind of joy that I needed.

2023 was a personal tabletop gold mine. We finally got Brennan Lee Mulligan’s first official 5e homebrew; I discovered tabletop meetup hub and game master meat
market StartPlaying.Games, where you can, as I have, hire a handsome Scottish man to D&D dominate you and your friends every week; and I played a ton of my new
favorite party game, As You ’Wich. It’s a dead-simple hand-builder that, for me at least, boils down to convincing your nephews to eat disgusting sandwiches
(hair, mustard, and broken glass on marble rye, absolutely not toasted).
</item 84>

<item 85>
Shohei Ohtani rumors tracker: Latest news, updates on free agent's MLB future (Sporting News, 2023-11-27T21:20:11+00:00)
Passan wrote a piece about 10 teams to watch the rest of the offseason, and he threw this little note in the section about the Red Sox, and how they needed to
sign Yoshinobu Yamamoto.

Boston's bats are good, or good enough at least, to avoid having to spend like mad in a down year for free agent hitters -- unless, of course, it's for Ohtani,
who has an affinity for Boston. If the Red Sox don't hit that jackpot, they're backfilled nicely with a system that includes outfielder Roman Anthony, shortstop
Marcelo Mayer and catcher Kyle Teel, among the best trios of offensive prospects in baseball.

It's cool when the best player in the world is a free agent and has an "affinity" for your town.

'Money's great, but winning's better'

Date: Nov.
</item 85>

<item 86>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
It delivers on the promise of building a hacker-samurai in a neon-infused open world replete with futuristic heists, daunting choices, and striking characters.
Phantom Liberty and Patch 2.0 may not completely erase the memory of CD Projekt’s initial botched release, but they come pretty damn close. Three years after we
first set foot in V’s shoes, Cyberpunk 2077 has finally justified the hype. —M. Mahardy

14. Dead Space

Developer: Motive Studio

Where to play: PlayStation 5, Windows PC, and Xbox Series X

With The Last of Us on HBO and Resident Evil 4 back in the conversation, it’s already a banner year for survival horror. Motive Studio’s Dead Space remake is no
exception. Following in the footsteps of Capcom’s aforementioned title, the original Dead Space brought the third-person-action focus of Resident Evil 4’s
formula to a deteriorating ship in outer space.
</item 86>

<item 87>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The book itself has a stunning and distinct art style that immediately transports you into the doom metal fantasy of the setting; but that’s only a small part
of its allure.

Unlike many TTRPGs that one merely has aspirations of playing one day, Mörk Borg does everything in its power to make the game easy to run. Although it’s a
short read, no line is wasted. Even item descriptions are dripping with hints at nefarious deeds one must have done to acquire them. The optional classes are an
absolute delight for those of us that savor the idea of playing the unsavory. From the esoteric introverts that choose to make their hovel far from prying eyes
to the unfortunate discarded-at-birth cutthroats who’ve had to scratch out a living any way they can, there is something for everyone. There is always the
option of creating something more tailor-made if none of the classes are appealing, but I appreciate that Mörk Borg offers beginner-friendly ideas and
archetypes to help you create something that fits right into the gritty setting.
</item 87>

<item 88>
Shohei Ohtani rumors tracker: Latest news, updates on Dodgers, Cubs & more as free agency short list emerges (Sporting News, 2023-12-01T22:05:46+00:00)
Imagine a lineup consisting of Ohtani, Ronald Acuna Jr., Ozzie Albies, Austin Riley and Matt Olson?

Sources: Braves are among the teams still engaged with Shohei Ohtani’s camp, as I reported in this @MLBNetwork segment. @MLB pic.twitter.com/lP4lISzttN — Jon
Morosi (@jonmorosi) December 4, 2023

Ohtani has already received several $500+ million offers

Date: Dec. 3

Source: Jon Heyman, NY Post

It's no surprise the bidding for Ohtani is heating up, with the star expected to sign soon. And New York Post's Jon Heyman reported Ohtani has already received
several deals that have topped $500 million, which could mean he winds up being the first $600 million player in MLB history. He added there are believed to be
five teams in the running for Ohtani.
</item 88>

<item 89>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Desperation

Tim Hutchings, game designer and academic

The best game I’ve played this year is a tie between Desperation by Jason Morningstar and Orestes’ Choice by Susanne Vejdemo. Desperation is a card-driven TTRPG
with a marvelous mechanic: Players draw a card with a bit of spoken text on it and then assign those words to one of the characters in play. It is so dang smart
and makes for rich yet tight storytelling as the cards force new contexts for the actions of the characters.

The game comes with two dark American Gothic-flavored scenarios: a starving Kansas town locked in a brutal winter and a cursed fishing boat lost in the north
Pacific. The faux woodcuts by Jabari Weathers and Brennen Reece are wonderful to look upon and convey the game’s tone as much as the flavorful card text.

My best game experience was Orestes’ Choice, a LARP by Susanne Vejdemo.
</item 89>

<item 90>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
It is up to you to go on a perilous quest of your own to retrieve your master’s fabled weapon in time for their final duel.

Throughout this transformational quest, you use three six-sided dice to resolve mechanics for racing against time, overcoming treacherous encounters, and
uncovering new truths about your master’s shrouded past. Embrace your inner doggy, buy this 17-page game, and begin your adventure today! Swiftly now — your
master’s fate is in your paws!

Home

Wesley “Nox” Crowe, aka Noxweiler Berf, creative director for Hunters Entertainment

Doug Levandowski and Yeonsoo Julian Kim’s Home drew me in immediately, and has been one of my most recommended tabletop role-playing games. If you know me or my
work, then the fact that it’s a horror game probably won’t come as much of a surprise. It’s a haunted house exploration simulator that can be played with a full
table of players or alone as a solo journaling experience.
</item 90>

<item 91>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
In the vein of Event Horizon, Sunshine, and Alien, Dead Space was a paragon for sci-fi horror in a confined and claustrophobic setting. Its remake has brought
that same vision to gorgeous new life, bringing quality-of-life changes and underappreciated updates (it has made several previously useless weapons into viable
tools in protagonist Isaac Clarke’s arsenal), making it hard to imagine ever going back to Visceral Games’ phenomenal original. —M. Mahardy

Related The Dead Space remake changes all the right things

13. Octopath Traveler 2

Developers: Square Enix, Acquire

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, and Windows PC

The first Octopath Traveler was one of those games that was as enjoyable to play as it was painful: enjoyable because so much of it kicked ass, but painful
because so much of it dragged the positive aspects down. In other words, it stood on the precipice of excellence, but couldn’t quite cross the line.
</item 91>

<item 92>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Players take on the role of soldiers and other hangers-on just trying to make it out alive. The heady mix of dice-driven gunplay and a playing-card-style
encounter creation makes it easy to run. But the fun begins even before that with character generation, which uses old school randomized tables to simulate an
entire military career. If you’re looking to get your video gaming buddies around the table for something that doesn’t require a headset or shouting down
pre-teens in a glitchy public lobby, the all-inclusive starter set comes highly recommended.

Project ECCO

It’s a near-impossible task to get a consistent group together, so solo TTRPGs have become my primary method of gaming. However, the barrier for me is finding a
game with a balance of narrative and mechanics that isn’t overwhelmingly crunchy, but also isn’t open-ended to the point of choice paralysis. For me, Project
ECCO found that balance.
</item 92>

<item 93>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game improves upon Star Wars Jedi: Fallen Order in every conceivable way, with a more entertaining, action-oriented start and a ton of aerial movements and
lightsaber stances that make you feel even more like a seasoned Jedi Knight. Quality-of-life improvements like fast travel make the game less frustrating to
play, while the new locales packed with hidden collectibles and upgrades make exploration more rewarding.

But what makes Jedi: Survivor truly special isn’t this crude matter, but its luminous heart. You feel it in the memorable characters you meet on your galactic
journey, be it the people you help or the friendships you forge and reconcile with. It’s in the classic, crowded cantina where you actually want to go check in
with the barkeep. It’s in the tactile, reverent way you craft your lightsaber. And it’s in the kinetic set-pieces that remind you of the serials Star Wars was
originally inspired by. At its best, you really can feel the Force around you. —CA

31.
</item 93>

<item 94>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Our campaign is called The Heartsguard Saga, and it’s based off Deven Rue’s map of a place called Euphoros. My players are at around level 15 and I frequently
give them a variety of amazing and unique gear that I’ve dreamed up all on my own... which they go on to forget they have, and never use. We used to play in
person, but since the pandemic I run the game virtually online from my house. I put a camera hovering over my gaming mat, and make a labeled grid for the
players so they call out, “Move me from A5 to B10.” During important rolls, I even make my players face their camera toward their dice for added suspense!

Petrana Radulovic, entertainment reporter

I’d wanted to play Dungeons & Dragons since I was in high school, but I never knew how to get into it.
</item 94>

<item 95>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Galaxy-spanning faction quests give you plenty to do, but it’s the side quests that really make you feel like the captain in something Star Trek adjacent:
generation ships and superhero hideouts and colonies of clones. There’s good sci-fi here for those willing to make the journey.

It’s true that the game oversold the idea of a galaxy with over 1,000 planets, a number that pales in comparison to the game’s most obvious competitor, No Man’s
Sky. It’s a shame this became so much of the focus thanks to Bethesda’s marketing, because that same focus obscured the absolutely fascinating new game plus
mode at the game’s heart. Consider this: No Man’s Sky is the game that realizes the universe’s infinite number of planets, while Starfield realizes the
universe’s infinite number of possibilities. —Clayton Ashley

33.
</item 95>

<item 96>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I create my custom protagonist, determining stats and bonuses based on my backstory, home world, and so on. I’m then thrust into a high-pressure job interview
as a potential heir to a Rogue Trader aboard her flagship. Little do we know that there’s a coup in the works, and traitors aboard the ship. After a deadly
struggle against heretics and demons, I ascend to the position of Rogue Trader. I’m the captain now, and I get to decide how to run my ship.

This is a vast game, with tons of features you’d expect from a CRPG — companions (each with their own narrative paths and conversation trees), top-down
strategic combat, and branching choices that impact the world around you. I get to make choices all day — in fact, that’s part of the gig of a Rogue Trader.
Some are more important than others; whenever I get to specific points in the story, I can choose from one of three major paths.
</item 96>

<item 97>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The gameplay didn’t just help me stick with the game, but instead allowed my excitement to bubble over every time I took on a new mission. —Ana Diaz

15. Cyberpunk 2077: Phantom Liberty

Developer: CD Projekt Red

Where to play: PlayStation 5, Windows PC, and Xbox Series X

Like the best DLC, Phantom Liberty does as much to expand on the base game as it does to reframe it. Alongside the sweeping Patch 2.0, which revamped Cyberpunk
2077’s role-playing progression systems, improved its enemy AI, and course-corrected a litany of other details, Phantom Liberty also adds a whole new district
to the dystopian world of Night City, complete with its own spy-thriller storyline.

The resulting Cyberpunk 2077 is not an entirely different beast than the one that was released in 2020, but it is a more evolved one.
</item 97>

<item 98>
MLB Winter Meetings tracker: Live updates on news, rumors, signings and trades for 2023 free agency (Sporting News, 2023-12-03T23:24:00+00:00)
Roberts said Ohtani is "clearly" the Dodgers' top priority, adding that the team is confident the two-time MVP will pitch again. Roberts wasn't able to offer a
timeline for Ohtani's decision.

Pair of pitchers could be signed after Ohtani deal

Date: Tuesday, Dec. 5

Source: Jon Morosi, MLB Network

MLB is largely in a holding pattern as it awaits news on Ohtani. MLB Network's Jon Morosi reported the Rays could wind up trading Tyler Glasnow shortly after
Ohtani signs, listing the Cubs and Reds as top contenders for the ace. He also reported Yoshinobu Yamamoto could sign following Ohtani's deal, and that the Red
Sox have him among their top priorities.

Cubs losing optimism in signing Ohtani

Date: Tuesday, Dec.
</item 98>

<item 99>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
Here’s what we found.

City of Winter

Linda Codega, Ennie Award-winning entertainment journalist

My friends and I still talk about the weekend we spent playing City of Winter. With lightweight rules but a heavy emphasis on the kind of ephemera usually
reserved for much crunchier games, creating space for this game is almost as important as who you chose to play it with. Characters travel along a cloth map
that illustrates a landscape with only barest indications of topography; the focus instead relies on themes of assimilation, diaspora, and — most importantly to
the narrative — baggage. As a family of immigrants (bound by blood or by choice), players describe their home, their traditions, and then, as a darkness comes
from the west side of the map, they leave it all behind, choosing what traditions and rituals they pick up from the places they visit and what they hold on to
from their homeland.
</item 99>

<item 100>
Rogue Trader’s role-playing embraces the brutality and freedom of Warhammer 40K (Polygon, 2023-12-06T13:00:00+00:00)
I can call on them to unlock doors or perform other environmental checks, but they come most in handy in combat.

When I meet opposition, it’s deeply satisfying to control my troops in a turn-based battle. Each fight takes place on a grid; it’s very similar to Baldur’s Gate
3 or even XCOM 2. Some positions provide cover, while others are out in the open. Friendly fire is also a very real concern. An arc of auto-fire from a bolter,
or a Navigator’s third eye opening, can harm friend as well as foe. My Rogue Trader is a sniper, and she would be lost without her Senechal taking the front
lines. There’s a lot of firepower at my disposal, and it’s mostly quite satisfying to use — even if I occasionally shred my poor Senechal with a devastating
AOE.

I can chat with the companions between battles, learning more about their pasts.
</item 100>
=== END ITEM id=c50c1179b63deb61 ===
