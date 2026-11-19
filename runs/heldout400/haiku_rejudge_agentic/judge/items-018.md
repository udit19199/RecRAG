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

=== ITEM id=1daa840c8e0729dd ===
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
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Insomniac’s 2018 installment, Grand Central
Station, Mr. Negative, Mr. Negative’s demons, hunters, Wake Up quest, taser, stun gun, bomb, Mary Jane, Essentially Sports
</item 1>

<item 2>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Spider-Man, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially
Sports
</item 2>

<item 3>
Graph query row:
from: Polygon article
to: Zazie Beetz
connection: ['The future of DC movies in 2022 and beyond', 'Netflix', 'Sandman greenlit for a second season', 'The future of DC movies in 2022 and beyond',
'Zazie Beetz']
relationships: ['MENTIONS', 'SAID', 'MENTIONS', 'MENTIONS']
evidence: [{'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nThe DC Extended Universe, as DC Comics’ movie and TV
line was originally called, has gone through some major growing pains since it launched with Zack Snyder’s Man of Steel in 2013. Looking for ways to compete
with Marvel Comics’ massively popular Marvel Cinematic Universe, DC has repeatedly switched tracks and tactics on the DCEU, sometimes steering toward a single
interconnected universe, then backtracking into a more director-driven plan where movies like Todd Phillips’ Joker could be made without fitting into any
specific previous continuity.\n\nIn 2023, the Snyderverse is coming to an end, slowly rolling to a halt with movies like Shazam! Fury of the Gods, The Flash,
and Blue Beetle, all leading up to the grand finale of Aquaman and the Lost Kingdom. Once the last bits of the DCEU make their way to theaters, it will be time
to usher in a new era of DC movies.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond
(Polygon, 2023-10-24T15:00:36+00:00)\nGuardians of the Galaxy and Suicide Squad director James Gunn and producer Peter Safran are taking the reins of the newly
branded DC Studios, and they’re bringing a long-term plan to DC with them. The new projects on the DC docket are meant to reboot or reshape DC’s best-known
heroes, while reaching into some deeper corners of the DC canon for more obscure characters. And it’s wrapping up the DCEU era, replacing it with just “the
DCU,” a mainline continuity that will exist alongside other DC film and TV universes. Here’s everything DC is currently planning to put in theaters and on TV
screens, in 2023 and well into the future.\n\n[Ed. note: This story only covers Warner Bros.’ DC projects and doesn’t include The CW’s Arrowverse, which is
effectively ending in 2023.]\n\n2023\n\nAquaman and The Lost Kingdom: Dec.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future
of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\n2023\n\nAquaman and The Lost Kingdom: Dec. 20\n\nExpect online fandom debates to go
ballistic again as we get closer to the sequel to James Wan’s joyously goofy 2018 under-the-sea extravaganza, Aquaman. Jason Momoa is back as the titular
superhero/sea king, with Nicole Kidman returning as his mother, Atlanna. Ben Affleck is reprising his role as Bruce Wayne/Batman, Patrick Wilson is back as
undersea usurper Orm the Ocean Master, Yahya Abdul-Mateen II is back as the villain Black Manta, and Wan is back at the helm. But most of the early noise about
Aquaman and the Lost Kingdom has been about the return of Amber Heard as Aquaman’s love interest, Mera — after the long, contentious legal battle between Heard
and Johnny Depp, expect the lead-up conversation about this one to be noisy.\n\n2024\n\nJoker: Folie à Deux: Oct.', 'article': 'The future of DC movies in 2022
and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\n2024\n\nJoker: Folie à Deux: Oct. 4\n\nTodd
Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with \u200b\u200bJoaquin Phoenix reprising
his role as Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love
interest, Sophie.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon,
2023-10-24T15:00:36+00:00)\nZazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.\n\nCreature Commandos\n\nJames Gunn is
listed as the writer for this seven-episode adult animated series, which will also include Suicide Squad head Amanda Waller. This time, she “creates a black ops
team out of monstrous prisoners.” The team will include Frankenstein’s monster (here called Eric Frankenstein and voiced by David Harbour), who has a long,
complicated DC history of his own. The series is set to star Frank Grillo (Prison Break) as Rick Flag Sr., Maria Bakalova (Guardians of the Galaxy Vol. 3) as
Princess Ilana Rostovic, Indira Varma (Game of Thrones) as the Bride of Frankenstein, and Alan Tudyk (Harley Quinn) as Doctor Phosphorus.', 'article': 'The
future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nWaller\n\nGunn’s
hilarious, heartfelt Suicide Squad spinoff series Peacemaker was greenlit for a second season in February 2022, but Gunn has now said he’s going to have to put
that project on hold for a bit to concentrate on other things — including Waller, the previously untitled Peacemaker spinoff centered on Viola Davis as
no-nonsense supervillain resource broker Amanda Waller. The character, introduced to DC’s movie continuity in David Ayer’s little-loved 2016 version of Suicide
Squad (though she’s been part of the comics continuity since 1986), is one of the few elements of Ayer’s movie that was held over in Gunn’s 2021
reboot/reshuffle movie, called The Suicide Squad. She also turned up for a quick cameo in Black Adam. Now she’s getting her own show — which “features Team
Peacemaker,” so it may feel like a stealth season 2, depending on who from the team winds up on the series.', 'article': 'The future of DC movies in 2022 and
beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\n2025\n\nSuperman: Legacy: July 11\n\nThe first of the
new Gunn/Safran projects to actually get a release date, Superman: Legacy seems expressly designed to break the DCEU mold. Henry Cavill is moving on from the
Superman role, and Gunn, who plans to script the film himself, made several pointed references to this film being a break from past visions of Superman: He
called the film “the true foundation of [their] creative vision for the DC Universe,” and said he “can’t wait to introduce [their] version of Superman.” The
advance description for this one currently just says it’ll feature the character “balancing his Kryptonian heritage with his human upbringing.”\n\nThe Batman
Part II: Oct. 3\n\nThe long, contentious history behind 2022’s The Batman didn’t leave much room for hope: As one of those movies that got bounced from team to
team for years, changing shape at every bounce, it looked like a project that was destined to be driven more by market calculation than by a specific vision.',
'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nBut
director and co-writer Matt Reeves found an effective take on the Dark Knight that resonated with audiences and landed solidly at the box office, and DC
immediately greenlit a sequel, again with Robert Pattinson as the Caped Crusader. No plot details are clear yet, though Pattinson and Reeves have been
blue-skying about a lot of things they’d like to do: The Court of Owls storyline, maybe? Mr. Freeze? Calendar Man, for god’s sake?\n\nUndated\n\nThe
Penguin\n\nHBO Max’s TV spinoff of Matt Reeves’ The Batman is meant to bridge the gap between that movie and Reeves’ sequel, as Penguin (Colin Farrell)
solidifies power in the Gotham underworld immediately after the death of Carmine Falcone (John Turturro) in the movie.', 'article': 'The future of DC movies in
2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nThe eight-episode series, produced by Reeves
and overseen by Agents of SHIELD writer-producer Lauren LeFranc, is meant as a lead-in for The Batman Part II, so even though it’s currently undated, expect it
before the movie hits.\n\nArkham Asylum\n\nThe actual title for the second of Reeves’ two HBO Max The Batman spinoffs hasn’t been revealed yet, but this one is
meant to be a prequel to the movie. Previously planned as a series about the workings of the Gotham police department, the series mutated in the development
stage, and Boardwalk Empire creator Terence Winter, originally in charge of the series, departed the project. Reeves has described this series as an origin
story for characters seen in Arkham in The Batman, with more of a horror movie or “haunted house” feel. The Staircase’s Antonio Campos is now the showrunner for
the series.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon,
2023-10-24T15:00:36+00:00)\nThe Staircase’s Antonio Campos is now the showrunner for the series.\n\nA ‘Black Superman’ movie\n\nDC announced in early 2021 that
writer Ta-Nehisi Coates would reboot Superman with a movie that would put a Black man in the title role. Details have been thin regarding whether this film
would tie into an existing DC comics story — for instance, whether this might be a movie outing for temporary Superman successor Steel or a complete reimagining
of the character — but the film was meant to be produced by J.J. Abrams under his Bad Robot imprint. Reportedly the project is still going forward in the wake
of Gunn and Safran’s new slate, but no release date has been announced.\n\nBrave and the Bold\n\nTaking its title from a silver age DC Comic, Brave and the Bold
will introduce Batman’s biological son Damian to the DCU — he’s the fifth of Batman’s many Robins — with a story inspired by Grant Morrison’s interpretation of
the character in the Batman and Son run of the Batman comic.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in
2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nThis will be Damian’s first appearance in a live-action film, though the character has appeared in
animated form in TV and movies. The continuity of this film will be separate from Matt Reeves’ planned The Batman trilogy; it’s meant to establish the new DCU
continuity for Batman and Robin.\n\nThe Authority\n\nThe WildStorm imprint — founded as an independent company in 1992 and bought up by DC in 1999 — maintained
its own superhero universe separate from DC’s, until DC shut it down in preparation for its characters to join the mainline DC Universe in the company’s 2011
reboot. In its day, Wildstorm published independent and creator-owned comics, alongside superhero series like Warren Ellis and Bryan Hitch’s The Authority,
about a group of seven heroes with high-concept superpowers, like the Engineer, a woman with nanotech blood, and Jack Hawksmoor, a man who psychically bonds to
cities. This will be the first appearance of WildStorm canon in live-action film.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The
future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nThis will be the first appearance of WildStorm canon in live-action
film.\n\nSupergirl: Woman of Tomorrow\n\nJust as Superman: Legacy is meant to present a new view of Superman for the DCU, Woman of Tomorrow is being billed as
“a science fiction adventure” that “presents a Supergirl viewers are not used to seeing.” The movie will be based on Tom King’s celebrated Supergirl comics, and
will presumably not touch on anything from the six-year run of Supergirl on The CW starring \u200b\u200bMelissa Benoist.\n\nSwamp Thing\n\nThe recent appearance
of Man-Thing in Marvel’s black-and-white special Werewolf by Night suggested there can be space for mainstream superhero comics’ occasional forays into EC
Comics-style “weird horror” territory. But it’ll be a while before we get to see what that looks like on the DC side.', 'article': 'The future of DC movies in
2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nSwamp Thing, a plant-elemental hero who
(depending on which comics you’re reading) might or might not be a mutated version of scientist Alec Holland, dates back to 1971, and was successfully rebooted
by Watchmen author Alan Moore in 1984. He’s had several previous screen incarnations, including Wes Craven’s 1982 movie Swamp Thing, a three-season USA Network
Swamp Thing series starting in 1990, and a little-watched, quietly dropped one-season series in 2019. DC Studios says this movie “will investigate the dark
origins of Swamp Thing.”\n\nBooster Gold\n\nDC hasn’t said much about this TV series, which brings the comics’ fan-favorite comic-relief character to the screen
for his own adventures. The company’s logline for the show is just “Booster Gold uses basic technology from the future to pretend to be a superhero in present
day” — which is pretty much Booster Gold’s traditional comics depiction as well.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The
future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nLanterns\n\nThe 2011 live-action Green Lantern movie (starring Ryan Reynolds as
original Green Lantern Hal Jordan) was such a legendary flop it scuttled DC’s plans at the time for a new interconnected superhero universe, and Reynolds keeps
finding new ways to publicly mock the film, especially in his Deadpool movies. This intergalactic-cop TV series will attempt to revive Hal Jordan for the
screen, teaming him up with later Green Lantern John Stewart to “uncover a dark mystery.”\n\nParadise Lost\n\nThis series is set in Wonder Woman’s home island
of Themyscira, and DC’s press release says it “focuses on the genesis and political intrigue of an island of all women.” Sounds a bit like The Lord of the
Rings: The Rings of Power, in terms of turning a familiar fantasy story into a political origin story. It’s unclear as of yet whether Wonder Woman herself will
have a role in the story, and whether the series will connect with Patty Jenkins’ Wonder Woman movies.', 'article': 'The future of DC movies in 2022 and
beyond'}, {'passage': 'The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nSandman season 2\n\nNetflix announced in November 2022
that its live-action series Sandman, based on Neil Gaiman’s seminal horror-fantasy comic, had been greenlit for a second season. No date has been announced yet,
and given the scale of the visual effects involved in the first season, it’s a fair bet that we won’t see the second season until late 2024 or early 2025 at the
earliest. Given that the first season covered the arcs of the first two Sandman graphic novels, Preludes and Nocturnes and The Doll’s House (with side stories
from the third volume, Dream Country), it’s a fair bet that season 2 will continue with book four, Season of Mists, the heavily foreshadowed confrontation
between Morpheus (Tom Sturridge) and Lucifer (Gwendoline Christie).', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC
movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)\nDead Boy Detectives\n\nIn spite of a series of cancellations and shake-ups at Max, Neil Gaiman
confirmed in January 2023 that the streamer’s Sandman comics spinoff Dead Boy Detectives was actively in production. But that February, the news broke that Max
had sold the series to Netflix, because it doesn’t fit with James Gunn’s plans for Warner Bros. DC content. The series is based on a pair of ghost boys, Charles
Rowland and Edwin Paine, who were introduced in Sandman’s Season of Mists storyline and later got their own 2001 comics miniseries. They also turned up in
season 3 of Max’s Doom Patrol series.', 'article': 'The future of DC movies in 2022 and beyond'}, {'passage': 'The future of DC movies in 2022 and beyond
(Polygon, 2023-10-24T15:00:36+00:00)\nThey also turned up in season 3 of Max’s Doom Patrol series. The initial press release on Dead Boy Detectives said it
would have an eight-episode first season, described as “a fresh take on a ghost story that explores loss, grief, and death through the lens of Edwin Payne
(George Rexstrew) and Charles Rowland (Jayden Revri), two dead British teenagers, and their very alive friend, Crystal Palace (Kassius Nelson).”\n\nPeacemaker
season 2\n\nAlongside everything else in this universe with his name on it, Gunn hasn’t forgotten Peacemaker. According to the director, Christopher Smith will
be making his return for season 2, sometime after Gunn wraps production on Superman: Legacy. This likely means that season 2 won’t be coming along until
sometime in 2025 or even later, but hey, at least we know it’s out there.', 'article': 'The future of DC movies in 2022 and beyond'}]
</item 3>

<item 4>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Wake Up, Good Men, Sable, Essentially
Sports
</item 4>

<item 5>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Insomniac’s team, Peter, Essentially
Sports, Miles
</item 5>

<item 6>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?

Graph facts:
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac -MADE-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Marvel’s Spider-Man 2, The Daily Bugle, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, PS5
</item 6>

<item 7>
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

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Xbox
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Insomniac
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Marvel’s Spider-Man 2
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> The Daily Bugle
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Spider-Man

Entities: Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, journalist, Peter, Essentially Sports, Spider-Man 2, Miles
</item 7>

<item 8>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.

Graph facts:
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Marvel’s Spider-Man 2
- Spider-Man 2 toys with Marvel history to be as much fun as possible -MENTIONS-> Insomniac
- The best video games of the year so far -MENTIONS-> Marvel’s Spider-Man 2
- Insomniac Games -MADE-> Marvel’s Spider-Man 2

Entities: Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Essentially Sports, Peter
Parker, Miles Morales, PS5
</item 8>

<item 9>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2024

Joker: Folie à Deux: Oct. 4

Todd Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with ​​Joaquin Phoenix reprising his role as
Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest,
Sophie.

Graph facts:
- The future of DC movies in 2022 and beyond -MENTIONS-> Colin Farrell
- The future of DC movies in 2022 and beyond -MENTIONS-> Ryan Reynolds
- The best movies of 2023 so far -MENTIONS-> Joaquin Phoenix
- The future of DC movies in 2022 and beyond -MENTIONS-> Netflix
- The future of DC movies in 2022 and beyond -MENTIONS-> DC Comics

Entities: Todd Phillips, Joker, Joker: Folie à Deux, King of Comedy, Joaquin Phoenix, Arthur Fleck, Lady Gaga, Harley Quinn, Zazie Beetz, musical thriller, The
future of DC movies in 2022 and beyond, Martin Scorsese, Sophie, DC, Polygon
</item 9>

<item 10>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?

Graph facts:
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Xbox
- Xbox -MENTIONS-> mobile market
- Sarah Michelle Gellar hits out at Roblox after her son’s account gets hacked -MENTIONS-> Xbox
- Microsoft Activision: What does deal mean for gamers? -MENTIONS-> Xbox
- The 53 best Black Friday deals we could find at Amazon, Walmart, Target and more -MENTIONS-> Xbox

Entities: Xbox, Insomniac, Marvel’s Spider-Man 2, Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way, Mary Jane, Starfield, Essentially
Sports, PlayStation, PlayStation 5
</item 10>
=== END ITEM id=1daa840c8e0729dd ===

=== ITEM id=4fcb4654cd1bebe5 ===
Gold sentences and facts:
Sentence 0: “Hopefully we can go one better this year,” he added, recalling the 2-0 defeat by Canada in 2022.
  Fact 0.0: The speaker said, 'Hopefully we can go one better this year.'
  Fact 0.1: The speaker recalled the 2-0 defeat by Canada in 2022.
Sentence 1: After a winless campaign in 2022, the Swans – led by ex-Collingwood player Chloe Molloy – have been competitive this season.
  Fact 1.0: The Swans had a winless campaign in 2022.
  Fact 1.1: The Swans are led by ex-Collingwood player Chloe Molloy.
  Fact 1.2: The Swans have been competitive this season.

Retrieved items:
<item 1>
Graph query row:
article: ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final
publisher: The Age
passage: ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.
connections: [{'related': 'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': 'South Africa vs Australia semi-final'}, {'related': 'Australia',
'relationship': 'PARTICIPATED_IN', 'entity': 'Australia vs South Africa Cricket World Cup 2023 Semi-Final Match'}, {'related': '‘Biggest win of my career’: De
Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'MENTIONS', 'entity': 'Australia'}, {'related': 'Davis Cup', 'relationship': 'MENTIONS',
'entity': 'Australia'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'PUBLISHED_BY',
'entity': 'The Age'}, {'related': 'Sydney Swans’ Academy', 'relationship': 'PART_OF', 'entity': 'Sydney Swans'}, {'related': 'Sydney Swans', 'relationship':
'PART_OF', 'entity': 'Sydney Swans’ Academy'}, {'related': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup
Finals and ODI series', 'relationship': 'MENTIONS', 'entity': 'India vs Australia 2023 World Cup final'}, {'related': 'Australia', 'relationship':
'PARTICIPATED_IN', 'entity': 'India vs Australia 2023 World Cup final'}, {'related': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia
match stats in World Cup Finals and ODI series', 'relationship': 'MENTIONS', 'entity': 'IND vs AUS World Cup 2023 Final: Weather Forecast, Pitch Report at
Narendra Modi Stadium Ahmedabad ahead of India vs Australia clash'}, {'related': 'India vs Australia 2023 World Cup final', 'relationship': 'MENTIONS',
'entity': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series'}, {'related': 'IND vs AUS
World Cup 2023 Final: Weather Forecast, Pitch Report at Narendra Modi Stadium Ahmedabad ahead of India vs Australia clash', 'relationship': 'MENTIONS',
'entity': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series'}, {'related': 'Australia',
'relationship': 'MENTIONS', 'entity': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI
series'}, {'related': 'India vs Australia match', 'relationship': 'MENTIONS', 'entity': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs
Australia match stats in World Cup Finals and ODI series'}, {'related': 'India vs Australia match in the ODI World Cup', 'relationship': 'MENTIONS', 'entity':
'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series'}, {'related': 'IND vs AUS
Head-to-Head Record in ODI World Cup: Check India vs Australia match stats in World Cup Finals and ODI series', 'relationship': 'MENTIONS', 'entity':
'Australia'}, {'related': 'South Africa vs Australia semi-final', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'Australia vs South
Africa Cricket World Cup 2023 Semi-Final Match', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'India vs Australia 2023 World Cup
final', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'India vs Australia match', 'relationship': 'PARTICIPATED_IN', 'entity':
'Australia'}, {'related': 'India vs Australia match in the ODI World Cup', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related':
'Australia-India Test', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'Australia loss to India', 'relationship': 'PARTICIPATED_IN',
'entity': 'Australia'}, {'related': 'Davis Cup final', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'Davis Cup', 'relationship':
'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'Davis Cup semi-final', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': '1991
Australia victory', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia'}, {'related': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs
Australia match stats in World Cup Finals and ODI series', 'relationship': 'MENTIONS', 'entity': 'India vs Australia match'}, {'related': 'Australia',
'relationship': 'PARTICIPATED_IN', 'entity': 'India vs Australia match'}, {'related': 'IND vs AUS Head-to-Head Record in ODI World Cup: Check India vs Australia
match stats in World Cup Finals and ODI series', 'relationship': 'MENTIONS', 'entity': 'India vs Australia match in the ODI World Cup'}, {'related':
'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': 'India vs Australia match in the ODI World Cup'}, {'related': 'Australia', 'relationship':
'PARTICIPATED_IN', 'entity': 'Australia-India Test'}, {'related': 'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': 'Australia loss to India'},
{'related': 'The Age', 'relationship': 'PUBLISHED_BY', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final'},
{'related': 'Australia', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final'},
{'related': 'Davis Cup semi-final triumph over Serbia', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power Australia
into Davis Cup final'}, {'related': 'Davis Cup win', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into
Davis Cup final'}, {'related': 'Davis Cup final', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into
Davis Cup final'}, {'related': 'Davis Cup round-robin matches', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power
Australia into Davis Cup final'}, {'related': 'Davis Cup knockout stage', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin
power Australia into Davis Cup final'}, {'related': 'Davis Cup', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power
Australia into Davis Cup final'}, {'related': 'Australian side', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin power
Australia into Davis Cup final'}, {'related': 'Davis Cup semi-final', 'relationship': 'MENTIONS', 'entity': '‘Biggest win of my career’: De Minaur, Popyrin
power Australia into Davis Cup final'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'relationship':
'MENTIONS', 'entity': 'Davis Cup semi-final triumph over Serbia'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup
final', 'relationship': 'MENTIONS', 'entity': 'Davis Cup win'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup
final', 'relationship': 'MENTIONS', 'entity': 'Davis Cup final'}, {'related': 'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': 'Davis Cup final'},
{'related': 'Australian side', 'relationship': 'PARTICIPATED_IN', 'entity': 'Davis Cup final'}, {'related': 'Davis Cup', 'relationship': 'PART_OF', 'entity':
'Davis Cup final'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'MENTIONS', 'entity':
'Davis Cup round-robin matches'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'MENTIONS',
'entity': 'Davis Cup knockout stage'}, {'related': 'Australia', 'relationship': 'MENTIONS', 'entity': 'Davis Cup'}, {'related': '‘Biggest win of my career’: De
Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'MENTIONS', 'entity': 'Davis Cup'}, {'related': 'Australia', 'relationship':
'PARTICIPATED_IN', 'entity': 'Davis Cup'}, {'related': 'Davis Cup final', 'relationship': 'PART_OF', 'entity': 'Davis Cup'}, {'related': '‘Biggest win of my
career’: De Minaur, Popyrin power Australia into Davis Cup final', 'relationship': 'MENTIONS', 'entity': 'Australian side'}, {'related': 'Davis Cup final',
'relationship': 'PARTICIPATED_IN', 'entity': 'Australian side'}, {'related': '‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup
final', 'relationship': 'MENTIONS', 'entity': 'Davis Cup semi-final'}, {'related': 'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': 'Davis Cup
semi-final'}, {'related': 'Australia', 'relationship': 'PARTICIPATED_IN', 'entity': '1991 Australia victory'}]
</item 1>

<item 2>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, The Age, West Coast, AFLW, Gold Coast, Geelong, Comeback kings Richmond take the AFLW points in Cairns, Western Bulldogs, Melbourne,
Cairns, Richmond, North Melbourne, Swans, Chloe Molloy, AFL Photos
</item 2>

<item 3>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The second of Sinner’s Djokovic victories came in Italy’s impressive Davis Cup semi-final triumph over Serbia, after saving three consecutive match points on
serve, before besting the world No.1 again in the doubles. World No.4 Sinner, coached by Australia’s Darren Cahill, is the first player to topple Djokovic twice
in the same season since Russia’s Daniil Medvedev in 2019. Italian teammates Lorenzo Sonego and Jannik Sinner will face Australia in the Davis Cup final.
Credit: Getty However, if Popyrin can get the job done in the No.2 singles then Australia will, at minimum, reach the doubles with a live chance, and a good
one, with 2022 Wimbledon champions Max Purcell and Matt Ebden to be unleashed.

Graph facts:
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Russia
- 'Too big to fail' Egypt faces economic crisis but has an anti-Gaza refugee policy. Israel and the U.S. hope that will change -MENTIONS-> Russia
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Russia
- Deepfake election risks trigger EU call for more generative AI safeguards -MENTIONS-> Russia
- Dissecting impact of Israel-Hamas conflict on markets -MENTIONS-> Russia

Entities: Russia, The Age, Australia, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Darren Cahill, Daniil
Medvedev, Lorenzo Sonego, Max Purcell, Matt Ebden, Italy, Davis Cup semi-final triumph over Serbia, 2022 Wimbledon championships, Popyrin, Davis Cup final
</item 3>

<item 4>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Credit: AP It had followed a superb win from teammate Popyrin, who defied jangling nerves to earn a 7-6 (7-5), 6-2 victory over Otto Virtanen. It meant the
Australians’ crack doubles team of Max Purcell and Matt Ebden weren’t even needed.

It will be Australia’s 49th appearance in the Davis Cup final, as they seek the trophy for the 29th t28-time champions Australia, in a 49th final, seek a first
win in the World Cup of men’s tennis in 20 years. “We’re a very solid squad, a lot of depth in every single position, and if we can get ourselves to a deciding
doubles, we’ve got an unbelievable pairing as well,” de Minaur said. “So we singles guys have just got do our best to try and get a win and let the doubles guys
do the job.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Australia, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Max Purcell, Matt Ebden, Popyrin, Davis Cup
final, De Minaur, Otto Virtanen
</item 4>

<item 5>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
The 22-year-old Sinner, by now an irresistible force, then re-emerged to team up with Lorenzo Sonego and spray massive forehand winners around liberally as they
outplayed a by now weary-looking Djokovic and Miomir Kecmanovic 6-3, 6-4. Djokovic had three match points in the singles rubber. Credit: Getty Images It was a
crushing end to Djokovic’s hopes of ending his triumphant season, which has featured three grand slam victories and an ATP finals showdown victory over Sinner
just six days ago, with a second Davis Cup title. After telling fans to “shut up” earlier in the week, he again got irritated with the crowd in the doubles,
pretending to be the conductor to the Italian fans’ deafening chorus, but though he may be the greatest player who ever picked up a racquet, doubles looks an
alien game to him.

Graph facts:
- Which M.L.B. Playoff Team Has the Most Daunting Home Field? We Ranked Them. -MENTIONS-> Getty Images
- Video games in 2023: Acquisitions, layoffs, unions -MENTIONS-> Getty Images
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Getty Images
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Getty Images
- Amazon sellers sound off on the FTC's 'long-overdue' antitrust case -MENTIONS-> Getty Images

Entities: Getty Images, The Age, Australia, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Lorenzo Sonego, Popyrin, Miomir
Kecmanovic, Davis Cup final, De Minaur, Sinner, Djokovic, singles rubber, doubles, ATP finals showdown, Italy
</item 5>

<item 6>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
It was something that I’ll never forget.” Loading Australia, who lost last year’s final to Canada, has enjoyed a soft run on paper to this year’s decider,
eliminating the Czech Republic and Finland – with neither country boasting a top-30 singles player – but will have earned their 29th Davis Cup title if they
account for Italy. Canada defeated Italy in the Davis Cup round-robin matches in September, but bowed out to Finland in the knockout stage without Felix
Auger-Aliassime and Denis Shapovalov, who have both battled injury and form. The final is due to start at 2am AEDT on Monday and will be broadcast on 9GEM.

Sinner ends Djoker’s Davis Cup dream Novak Djokovic’s dream of capping his great season by blasting Serbia past Australia in the Davis Cup final was blown up by
Jannik Sinner’s extraordinary double triumph over the world No.1.

Graph facts:
- Canada vs Jamaica live score, updates, highlights, result from CONCACAF Nations League quarterfinal second leg -MENTIONS-> Canada
- Jamaica vs Canada live score, result, updates, highlights from CONCACAF Nations League quarterfinal first leg -MENTIONS-> Canada
- Ismael Kone -WORKS_FOR-> Canada
- Biello -WORKS_FOR-> Canada
- Mauro Biello -WORKS_FOR-> Canada

Entities: Canada, The Age, Australia, Novak Djokovic, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Italy,
Popyrin, Davis Cup final, Davis Cup round-robin matches, Davis Cup knockout stage, Serbia, De Minaur, Felix Auger-Aliassime, Denis Shapovalov
</item 6>

<item 7>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.

Graph facts:
- Livingstone -SAID-> Women’s World Cup
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Western Bulldogs, Molloy, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds,
Swans, AFLW competition, game time and scoring, AFL, The Guardian
</item 7>

<item 8>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.

Graph facts:
- West Indies v England: first T20 international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> West Indies v England: first T20 international
- Livingstone -SAID-> game time and scoring
- West Indies v England: fifth and deciding men’s T20 cricket international – live -MENTIONS-> Livingstone
- Livingstone -PARTICIPATED_IN-> first T20 international

Entities: Livingstone, Jack Snape, AFLW, Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Swans, Matildas, domestic
women’s sport, long-term growth, Perth, AFL, Sydney, The Guardian, Women’s World Cup
</item 8>

<item 9>
‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final (The Age, 2023-11-24T23:10:22+00:00)
Instead, in the Davis Cup final in Malaga, Australia will face an Italy team running hot on the heroics of their golden boy Sinner, who beat Djokovic in both
the singles and the doubles in the space of five exhilarating hours in their semi-final on Saturday (Sunday AEDT). Jannik Sinner saved three match points before
going on to beat Novak Djokovic. Credit: Getty Images In one of the great Davis Cup rubbers, Djokovic had held three match points against Sinner and was on the
verge of a completing a relatively workaday 2-0 victory for the Serbs.

But for only the fourth time in his unparalleled career, Djokovic was beaten from match-point up as Sinner roared back to complete his second victory over the
master in just 12 days, winning 6-2, 2-6, 7-5.

Graph facts:
- How ego and fear fuelled the rise of artificial intelligence -PUBLISHED_BY-> The Age
- ASX set to open higher as Wall Street rebounds; $A rises -PUBLISHED_BY-> The Age
- Is Google Search better than the rest? And is that fair? -PUBLISHED_BY-> The Age
- When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover -PUBLISHED_BY-> The Age
- At 37, Alexandra found herself broke, heartbroken and living with her parents -PUBLISHED_BY-> The Age

Entities: The Age, Australia, Novak Djokovic, Jannik Sinner, ‘Biggest win of my career’: De Minaur, Popyrin power Australia into Davis Cup final, Popyrin, Davis
Cup final, De Minaur, Italy team, Serbs, Malaga, Davis Cup semi-final
</item 9>

<item 10>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.

Graph facts:
- Ground-breaking Gold Coast: Suns make AFLW finals for a second time -MENTIONS-> Adelaide
- Comeback kings Richmond take the AFLW points in Cairns -MENTIONS-> Adelaide
- NBL Round 5: Harvey hits Hail Mary for Hawks to topple Taipans at the buzzer, Cheatham injured as Sixers upset Breakers -MENTIONS-> Adelaide
- Najwa Allen -WORKS_FOR-> Adelaide
- Chelsea Randall -WORKS_FOR-> Adelaide

Entities: Adelaide, Jack Snape, Sydney Swans, AFLW, Henson Park, Loretta Johns, Australian rules football, Harbour City, marquee pride match, Craft beer and
cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds, Collingwood, Sydney, The Guardian
</item 10>
=== END ITEM id=4fcb4654cd1bebe5 ===
