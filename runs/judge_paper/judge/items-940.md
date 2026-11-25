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

=== ITEM id=0006d5b60d468a14 ===
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
Mary’s gameplay missions end up increasing the overall quality of Spider-Man 2. All things considered, it is great to see Insomniac improving on this aspect
instead of removing it completely.

ADVERTISEMENT Article continues below this ad

Marvel’s Spider-Man 2 is now available on PlayStation 5 worldwide.

Watch This Story: The Console Wars Are Heating Head-to-Head: Xbox and PlayStation Fans Battle Over Which Custom Controller is Better, Starfield or Spider-Man 2?
</item 1>

<item 2>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
It’s an absolute miracle Scorsese and Schoonmaker manage to produce a masterpiece out of the chaotic pieces of this production process (again in post, with
ample voice-over). Everyone is letting loose, and to hear the actors and the director describe the shoot after the fact, it takes on something of a religious
quality, with Scorsese as the wrinkled shaman handing out the peyote.

What grounds it is the framing. Scorsese is telling us a familiar story we’ve come to expect from him, about criminals and the pleasure they take in committing
crimes and reaping the benefits. The brilliance is the way he mines filmgoers’ long relationship with his work, repurposing the style and tone of his gangster
classics and inserting it into the world of finance. As he’s done effectively since Casino, Scorsese weaponizes postmodernism, using viewers’ presumed existing
familiarity with his films as context.
</item 2>

<item 3>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on AMC Plus

Sympathy for the Devil

Where to watch: Available to stream on AMC Plus

Genre: Psychological thriller

Run time: 1h 30m

Director: Yuval Adler

Cast: Nicolas Cage, Joel Kinnaman

A largely two-person movie that sounds a bit like Collateral, Sympathy for the Devil stars Nicolas Cage as a passenger who holds a driver (Joel Kinnaman)
hostage on a long car trip.

New to rent

Paw Patrol: The Mighty Movie

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Action adventure

Run time: 1h 35m

Director: Cal Brunker

Cast: Mckenna Grace, Taraji P. Henson, Marsai Martin

The Paw Patrol is back — this sequel to the first movie sees the pup get superpowers in their quest to stop Mayor Humdinger from destroying Adventure City.
</item 3>

<item 4>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
As we donned Samus’ suit and explored strange planets, aggressive alien lifeforms could now get right in our faces, forcing us to dodge, strafe, and roll (in
morph ball form, naturally) using all three dimensions. No longer would we sit back and watch as Samus dipped her toe into a pool of lava; in first-person, as
molten fire spread over our visor, we’d really feel the pressure to find that Varia Suit upgrade. And perhaps most importantly, from behind Samus’ visor, we
gained the ability to scan our enemies and environment, collecting and translating logs from the long-dead Chozo aliens who once inhabited these now-hostile
places.

The world of Prime is harsh and unrelenting. (Save points will, at times, be quite far from one another.) But it’s worth buckling down and pushing through the
pain points to discover this world’s secrets. —M. Myers

18.
</item 4>

<item 5>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
They haven’t made any claims of breakup and have been silent about their relationship with each other. The only things that have changed are Savannah’s
Instagram feed. She has stopped posting pictures with Danis, which were quite prevalent back in 2020. Also, another thing that hints towards their breakup is
Danis’s first press conference for the Logan Paul fight.

ADVERTISEMENT Article continues below this ad

During the fight, Logan Paul brought up the topic of Montano, when Danis pointedly mentioned their breakup. Paul said, “Yo you’ve been so salty ever since
Savannah rejected your a** to go hook up with real athletes. You don’t gotta take that s**t out on me because my brother hooked up with your ex.” Danis scoffed
at Maverick’s claim and clapped back, “At least I got rid of the sl*t, you’re marrying one.”
</item 5>

<item 6>
Video games in 2023: Acquisitions, layoffs, unions (Engadget, 2023-12-19T14:30:37+00:00)
Microsoft owns nearly 40 developers and it bought a chunk of those in the past five years. The Xbox umbrella covers 343 industries, Arkane Studios, Bethesda,
Compulsion Games, Double Fine Productions, id Software, Infinity Ward, Mojang Studios, Ninja Theory, Playground Games, Tango Gameworks and Turn 10, among dozens
more.

Charley Gallay via Getty Images

With these studios at its back, Microsoft is leaning hard into cloud gaming while attempting to build a device-agnostic ecosystem powered by the Xbox brand.
These moves are designed to unlock the mobile market even more, putting Xbox games on all devices, everywhere, all the time.

Still, Sony is bigger than Microsoft by revenue. Though Microsoft is often the face of the game-studio acquisition spree, Sony is the owner of 21 development
teams, including Bungie, Guerrilla Games, Haven Studios, Housemarque, Insomniac Games, Media Molecule, Naughty Dog and Sucker Punch Productions.
</item 6>

<item 7>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
(If you can get past the Frankenstein-ification of the boroughs, that is. RIP Chelsea.)

There are hints of greater thematic aspirations within Spider-Man 2. Questions about mass incarceration, recidivism, and reintegration into society for formerly
incarcerated people; worries about the surveillance state and what we give up when we engage with modern technology. These things appear, however fleetingly, in
Spider-Man 2, but there isn’t enough time to elaborate on any single one in the unrelenting march of set-piece after glorious set-piece.

But I’m glad they’re there, anyway. Because if we’re going to retell a story, we ought to retell it anew. Canons were made to be broken, and Spider-Man 2 swings
in that direction. With a sequel teed up by the game’s final act, I’m eager to see where Insomniac goes with that momentum. Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct.
</item 7>

<item 8>
The best shows to stream in December (The Sydney Morning Herald, 2023-11-30T22:00:00+00:00)
With Juliette Lewis, Melanie Lynskey and Christina Ricci among the survivors, the story is a mix of supernatural menace, black comedy, teenage adversity, and
deeply expressed trauma. It’s coming to Netflix after two years on Paramount+ and it’s a can’t-look-away binge just waiting to happen. When the history of the
high-concept Hollywood blockbuster is written, Zack Snyder’s name will be prominent – for better or worse. Snyder is responsible for 300, Man of Steel, and most
of Justice League, and having come to Netflix for the tepid zombie heist movie Army of the Dead he’s doubling down with Rebel Moon: Part One – A Child of Fire
(December 22).
</item 8>

<item 9>
Every movie and show coming to Netflix in November (Polygon, 2023-11-01T12:00:00+00:00)
Well, now you do!

I could talk about how with each passing year, David Fincher’s Mark Zuckerberg biopic grows increasingly more relevant, as Facebook’s (excuse me, Meta’s) grip
on the internet becomes stronger, but that’s been done to the death. Instead, I’m going to praise both Jesse Eisenberg and Andrew Garfield for their electric
performances and their zinging chemistry. Come for the Facebook and Mark Zuckerberg origin story, stay for the tale of doomed friendship (and maybe write a
fanfic about it?).

Also, this is a great November movie because of the autumnal shots of Harvard and the Winklevoss twins rowing. Impeccable vibes to fuel the dark academia
aesthetic TikTok videos.

Available Now

Slayaway Camp 2: Netflix & Kill (Netflix Games)

From Netflix: Leave no survivors. In this sequel, you’re Skullface, a killer stuck inside a retro horror movie catalog — solve puzzles, stalk victims and get
stabby.
</item 9>

<item 10>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
We don’t dig into the considerable lore behind FNAF here — there are endless YouTube videos for that. We’re mostly talking about who this film is for and what
to expect.

Does Five Nights at Freddy’s have a post-credits scene?

Yes. More specifically, it has a brief mid-credits scene that rolls less than a minute after the initial cut to black. It isn’t integral to the movie’s lore,
and doesn’t feel like a sequel tease, though it certainly could be elaborated on in a sequel. It’s more of a comic scene involving YouTuber and horror-game
aficionado Cory DeVante “CoryxKenshin” Williams, though he doesn’t seem to be playing himself. But as far as a true post-credits scene that actually plays at
the tail end of the credit roll, Five Nights at Freddy’s doesn’t have one — just a brief, not especially exciting audio-only Easter egg.

Can you enjoy Five Nights at Freddy’s if you haven’t played the games?
</item 10>

<item 11>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
MJ in Spider-Man 2 vs. MJ from Insomniac’s 2018 installment

While there are similarities between the character’s iterations in two titles, a lot of drastic differences are also present. Speaking of similar aspects, the
first game had a mission where Mary gets to use a taser to shock the guards. Well there’s no taser in Spider-Man 2, she’s got a stun gun now.

Then, there are moments of hiding and sneaking past the enemies. In the original game, there were missions like the one in Grand Central Station where MJ had to
stay hidden from a number of Mr. Negative’s demons to defuse a bomb. Similarly, in Spider-Man 2, MJ is seen hiding from the hunters during the ‘Wake Up’ quest.
But what’s actually changed?

Well, the very difference lies in the lack of MJ’s abilities in the first game and how she’s evolved in the second.
</item 11>

<item 12>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to break free of expectation and canon

Narratively, juggling two Spider-Mans requires a different kind of deftness. Two villains new to the series take center stage — Kraven the Hunter and Venom — as
well as a returning cast of nearly every villain from the previous two games. Without delving into spoilers, a large majority of that cast is sidelined early
on, leaving Kraven and Venom as our principal antagonists, albeit with large roles for returning villains the Lizard (who made a brief appearance in Miles
Morales) and Mister Negative. There are others besides them that should delight comic fans, setting up future conflicts for either expansions or the presumed
Spider-Man 3 to tackle. If what you come to Spider-Man for is spectacle and villainy, Spider-Man 2 has the goods and then some.

As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up.
</item 12>

<item 13>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
The Staircase’s Antonio Campos is now the showrunner for the series.

A ‘Black Superman’ movie

DC announced in early 2021 that writer Ta-Nehisi Coates would reboot Superman with a movie that would put a Black man in the title role. Details have been thin
regarding whether this film would tie into an existing DC comics story — for instance, whether this might be a movie outing for temporary Superman successor
Steel or a complete reimagining of the character — but the film was meant to be produced by J.J. Abrams under his Bad Robot imprint. Reportedly the project is
still going forward in the wake of Gunn and Safran’s new slate, but no release date has been announced.

Brave and the Bold

Taking its title from a silver age DC Comic, Brave and the Bold will introduce Batman’s biological son Damian to the DCU — he’s the fifth of Batman’s many
Robins — with a story inspired by Grant Morrison’s interpretation of the character in the Batman and Son run of the Batman comic.
</item 13>

<item 14>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
The possessed-animatronics horror video game Five Nights at Freddy’s is less than a decade old, but a movie version has been in the works almost since the game
launched, which seems especially protracted for a development process. Warner Bros. bought the movie rights in 2015, and the project moved over to Blumhouse a
couple of years later. Game creator Scott Cawthon wrote and rewrote the script, while kid-horror filmmakers like Gil Kenan (Monster House) and Chris Columbus
(Gremlins, the first Harry Potter and Percy Jackson movies) signed on and jumped off the project.

Finally, under director Emma Tammi, the film is hitting theaters simultaneously with a Peacock streaming debut. The franchise’s most hardcore fans probably
consider themselves beyond ready to watch the Freddy’s film. But what about everyone else? If you’re not sure, have a look at this handy (and extremely
spoiler-light) Q&A.
</item 14>

<item 15>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
While the playing habits of a Twitch streamer don’t represent the average player by any means, content creators certainly shape public opinion about games.

“Negative sentiment is very profitable,” says the AAA game developer, who noted that social media platforms are built to highlight popular — and sometimes
caustic — content. This dynamic sets up an inherent tension between game developers and hardcore players like Twitch streamers, who are not the main audience of
most games but are often some of the most visible players.

“I don’t at all think that streamers are embodying some typical player behaviors, like ‘Brandon,’” the developer continues, posing a name for an imaginary
everyday consumer. “There’s definitely a model of consumers nowadays who probably parrot those sentiments. They did not arrive at those conclusions themselves.”

“There is inherently [a] vicious and negative cycle for everyone involved, for how live service interacts with Twitch as a medium”

The misinformation will vary, but Twitch both shapes that discourse while also becoming the very metric that people use to make sense of a game’s situation.
</item 15>

<item 16>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
Things haven’t changed much since 2019 at some major studios: Diablo 4’s developers say they also underwent crunch to ship the game. Naturally, the action
role-playing game had Twitch integration from the jump, and alongside a detailed content roadmap. There’s little incentive for any of this to change, even as
big-budget games buckle under their own weight and studios are closing left and right. Not when publishers know that the longer a player is hooked, the more
likely it is that they’ll spend money.

At this point, live-service games need content creators like Twitch streamers to stay solvent. According to the developer speaking to Polygon, streamers help
old games transform into new and visible moments. Developers need live-service games to stay in the conversation, because the base experience can worsen without
a persistent player base.

It’s “not even from a cost perspective,” the developer says, though obviously more players make profits more viable. “But [if] we don’t have 20,000 players,
then matchmaking times are terrible.
</item 16>

<item 17>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
If you want curated lists of our favorite media, check out What to Play and What to Watch .

Retellings are compelling because they allow writers to toy with readers’ preconceived notions of how the story “should go,” both in terms of genre conventions
and in terms of plot. We know Little Red Riding Hood will go to her grandmother’s house, where she will find that her grandmother has been eaten by a wolf, just
as we know Spider-Man will be bitten by a radioactive spider. The fun, both for the author and the reader, comes in deviating from those norms — in finding new
ways to tell old tales.

With Marvel’s Spider-Man 2, Insomniac Games attempts something similar.

Mainstream Spider-Man stories are increasingly Spider-Men stories, and Spider-Man 2, the third Spidey game from Insomniac, is no different.
</item 17>

<item 18>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
This will be the first appearance of WildStorm canon in live-action film.

Supergirl: Woman of Tomorrow

Just as Superman: Legacy is meant to present a new view of Superman for the DCU, Woman of Tomorrow is being billed as “a science fiction adventure” that
“presents a Supergirl viewers are not used to seeing.” The movie will be based on Tom King’s celebrated Supergirl comics, and will presumably not touch on
anything from the six-year run of Supergirl on The CW starring ​​Melissa Benoist.

Swamp Thing

The recent appearance of Man-Thing in Marvel’s black-and-white special Werewolf by Night suggested there can be space for mainstream superhero comics’
occasional forays into EC Comics-style “weird horror” territory. But it’ll be a while before we get to see what that looks like on the DC side.
</item 18>

<item 19>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Maybe Hollywood as a whole took away the wrong lesson (we’ve seen so many multiverse movies), but Into the Spider-Verse redefined genre staples.

The sequel continues to raise the bar. It’s a little busier than the first, pulling in more characters, more frantic visuals, more emotional throughlines, but
it’s still bold, daring, and unlike anything else in the superhero movie sphere.

As the characters bounce across the multiverse, the animation style shifts, with each character rooted in a different style. Spider-Gwen’s world is rendered
with watercolors, which shift according to the tone of her story. She gets a step up in this movie — a second chance for Gwen Stacy, whose previous iterations
across media have not ended well. Meanwhile, Spider-Punk, the brash counterculture rebel, is made up of collaged bits, and animated against the norm so he
stands out even more. Every frame of the movie is a visual feast, just bursting with lovingly rendered details.
</item 19>

<item 20>
The best new TV series on every streaming service (Polygon, 2023-10-25T16:09:11+00:00)
While Taika Waititi produces the show, it is blissfully removed from his particular (and for me, very tiring) brand of humor, instead shining through
award-winning filmmaker Sterlin Harjo and a dynamic cast of main characters and featured guests (the second season included a memorable guest appearance by Prey
star Amber Midthunder, and the legendary Gary Farmer and Wes Studi have hilarious recurring roles). The third season wrapped the show up on its terms, leaving
an indelible mark on television and the culture at large. —PV

Also good: Love Island UK, the reality dating show, What We Do in the Shadows, the hilarious TV adaptation of the modern-day vampire mockumentary, and the final
season of Atlanta, the “Twin Peaks for rappers” comedy drama starring (and created by) Donald Glover.

The best new TV on Prime Video

Gen V

Showrunners: Michele Fazekas, Tara Butters

Cast: Jaz Sinclair, Chance Perdomo, Lizze Broadway

Becoming a superhero is hard.
</item 20>

<item 21>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
Like everything else in the Flower Kingdom, a few enemies shift and change in bizarre ways when the Wonder Flower is activated, like Hoppo, the bouncy, round
hippo equivalent that grows in size to create chaos in one of Super Mario Bros. Wonder’s many mesmerizing levels.

All of this weirdness, combined with Nintendo’s platforming expertise, makes Super Mario Bros. Wonder an entrancing romp alone, with friends, or with strangers
online. This is the freshest Mario has felt in decades, and it bodes well for the iconic plumber’s 2D future. —N. Carpenter

3. Alan Wake 2

Developer: Remedy Entertainment

Where to play: PlayStation 5, Windows PC, and Xbox Series X

There’s a body in the forest with a hole in its chest. The forest, like the body, is also missing its heart, and a woman must find both. In each, there are
pages. This is a story, broken into pieces, we learn.
</item 21>

<item 22>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Every frame of the movie is a visual feast, just bursting with lovingly rendered details.

With a bigger cast and higher stakes, Across the Spider-Verse wasn’t going to finish its story in just one movie. It ends on a huge cliffhanger, but all the
pieces are in place for one hell of a trilogy finale. —PR

7. The Boy and the Heron

Director: Hayao Miyazaki

Cast: Soma Santoki, Masaki Suda, Ko Shibasaki

Where to watch: Theaters

Texture is essential to Hayao Miyazaki’s work. Sometimes the most visceral memories of his movies are the ways liquids glob or glide, while wood flecks flint
and flake in tiny explosions. Famously, even his still shots contain some motion, often featuring the slightest of movements that make the world feel more full
— more real.
</item 22>

<item 23>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
It’s evidence DiCaprio hasn’t “just” been a muse and face on the posters; he’s had a say in deciding what projects Scorsese has taken on, and has even made
significant decisions in shaping Scorsese’s stories. (Scorsese entirely reimagined Killers of the Flower Moon after DiCaprio offered notes on an early draft of
the script, including shifting focus to a different main character and moving DiCaprio into that role.)

For certain older filmmakers — say, Clint Eastwood — aging has meant settling for more routine and less challenging stories and shots, on a tight production
schedule. Scorsese went in the other direction. From the outside, The Departed appears to be a cynical, domestic remake of a Hong Kong cops-and-robbers
shootout. But Scorsese makes it his own and breathes bizarre life and energy into it. It contains the continuity errors, quick cutting, both rhythmic and
arrhythmic needle bombs, and goofball humor his movies hadn’t featured since Goodfellas.
</item 23>

<item 24>
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
</item 24>

<item 25>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
One can even call her the third unofficial superhero in the latest PS5 exclusive.

A refreshing take on Mary Jane’s stealth missions makes her a strong character in Marvel’s Spider-Man 2

ADVERTISEMENT Article continues below this ad

In the first title, stepping into MJ’s shoes was a great concept, but the devs somehow couldn’t execute the plan flawlessly. As a result, missions featuring her
as a playable character were heavily criticized by most of the fans. There was nothing too special about those quests whatsoever.

Now, looking at Mary Jane in Marvel’s Spider-Man 2, there’s a very clear difference one can make out. In short, it feels comparing night and day. Being a
journalist at The Daily Bugle, her character is bound to go places in order to investigate mysteries. The first game had such situations, but the sequel
improved on them. It gives MJ a lot more strength instead of being someone who always needs to be rescued in the last moment. But how?
</item 25>

<item 26>
Microsoft Activision: What does deal mean for gamers? (BBC News - Technology, 2023-10-13T23:01:24+00:00)
And critically, Microsoft has a habit of making its big blockbuster games available to play on Game Pass, for a monthly fee, on the first day they release.

Image source, EPA-EFE Image caption, Bethesda, which is owned by Microsoft, launched its new game Starfield in 2023 - but only on Xbox and PC

It's up to gamers to decide for themselves whether it's a better deal to buy the game or pay to access it.

But it is worth noting this concerned the UK regulator so much it made sure Activision Blizzard games couldn't be streamed exclusively on Xbox for 15 years.

That might be because Microsoft has form here - when the hotly anticipated Starfield came out in September 2023 to critical acclaim, it was available
exclusively on PC and Xbox, and playable on Game Pass from day one.

If games are still on PlayStation, what's in it for Microsoft?

Even with games still able to appear on PlayStation, it's Microsoft who will be winning out.
</item 26>

<item 27>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Elizabeth comes with an open mind, ready to inhabit Gracie and fully understand her. And yet, across Todd Haynes’ May December, we see how nothing is as simple
as Elizabeth — or the audience — thinks. Gracie is always more than meets the eye: more emotional, more aware, more monstrous. And throughout the movie, Haynes
expertly teases out the two women’s mysteries as they each try to craft an identity around the other.

Haynes’ film certainly has callbacks to other, classic movies about selfhood. But in a way, May December feels most in line with something like The Rehearsal,
which also has a lot on its mind about how blurry the line between real life and performance can be, and particularly how traumatic it is to ask kids to behave
like adults. It’s something Melton’s Joe is tragically caught in the middle of, and something May December always has on its mind, even when it feels like it’s
about other things.
</item 27>

<item 28>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
But director and co-writer Matt Reeves found an effective take on the Dark Knight that resonated with audiences and landed solidly at the box office, and DC
immediately greenlit a sequel, again with Robert Pattinson as the Caped Crusader. No plot details are clear yet, though Pattinson and Reeves have been
blue-skying about a lot of things they’d like to do: The Court of Owls storyline, maybe? Mr. Freeze? Calendar Man, for god’s sake?

Undated

The Penguin

HBO Max’s TV spinoff of Matt Reeves’ The Batman is meant to bridge the gap between that movie and Reeves’ sequel, as Penguin (Colin Farrell) solidifies power in
the Gotham underworld immediately after the death of Carmine Falcone (John Turturro) in the movie.
</item 28>

<item 29>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Swamp Thing, a plant-elemental hero who (depending on which comics you’re reading) might or might not be a mutated version of scientist Alec Holland, dates back
to 1971, and was successfully rebooted by Watchmen author Alan Moore in 1984. He’s had several previous screen incarnations, including Wes Craven’s 1982 movie
Swamp Thing, a three-season USA Network Swamp Thing series starting in 1990, and a little-watched, quietly dropped one-season series in 2019. DC Studios says
this movie “will investigate the dark origins of Swamp Thing.”

Booster Gold

DC hasn’t said much about this TV series, which brings the comics’ fan-favorite comic-relief character to the screen for his own adventures. The company’s
logline for the show is just “Booster Gold uses basic technology from the future to pretend to be a superhero in present day” — which is pretty much Booster
Gold’s traditional comics depiction as well.
</item 29>

<item 30>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Our best-of-the-year list reflects that range.

How the Polygon top 50 list works

Every year, the staff’s film fans create individual ballots to reflect their top movies, with an option to rank them numerically or just weight them. We use
these ballots to generate a collective list, weighted by strength of opinion to make sure a much-seen mainstream movie doesn’t have too much advantage over a
terrific but underseen indie. And then we put it all out there for your enjoyment. The top 10 picks on each our staff’s lists will be listed in the comments.

Any movie released in the US this calendar year is eligible, but since we are publishing this in early December, some December releases are underrepresented. We
hope you’ll find a new favorite here on our list of the best movies of 2023.

Honorable mentions Movies that received votes but did not crack our top 50: The Pope’s Exorcist, Bottoms, Rye Lane, No Hard Feelings, Hundreds of Beavers, Evil
Dead Rise, Guardians of the Galaxy Vol.
</item 30>

<item 31>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on Peacock

My Big Fat Greek Wedding 3

Where to watch: Available to stream on Peacock

Genre: Romantic comedy

Run time: 1h 32m

Director: Nia Vardalos

Cast: Nia Vardalos, John Corbett, Louis Mandylor

One of cinema’s most endearingly goofy families is back, in the most family-centric franchise this side of the Fast and Furious movies. It’s the first Greek
Wedding movie since 2016, which came nearly 15 years after the original smash hit. This time, star-writer Nia Vardalos takes over directorial duties, following
up her 2009 directorial debut I Hate Valentine’s Day.
</item 31>

<item 32>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
3, Nimona, The Starling Girl, Thanksgiving, Guy Ritchie’s The Covenant, The Eight Mountains, The Roundup: No Way Out, The Royal Hotel, Red, White & Royal Blue,
Blue Jean, The Venture Bros.: Radiant Is the Blood of the Baboon Heart, Quiz Lady, Air, Elemental, Extraction 2, Fair Play, Metalocalypse: Army of the Doomstar,
A Haunting in Venice, Fast X, You Hurt My Feelings, The Super Mario Bros. Movie, No One Will Save You, Monster, The Pigeon Tunnel, The Caine Mutiny
Court-Martial, Ant-Man and the Wasp: Quantumania, American Fiction, Maestro, Saw X, Maaveeran, The Creator, The Blackening, The Marvels, Afire, Kandahar,
Crater, Dark Harvest, Ernest & Celestine: A Trip to Gibberitia, Indiana Jones and the Dial of Destiny

Top 50

50.
</item 32>

<item 33>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Dead Boy Detectives

In spite of a series of cancellations and shake-ups at Max, Neil Gaiman confirmed in January 2023 that the streamer’s Sandman comics spinoff Dead Boy Detectives
was actively in production. But that February, the news broke that Max had sold the series to Netflix, because it doesn’t fit with James Gunn’s plans for Warner
Bros. DC content. The series is based on a pair of ghost boys, Charles Rowland and Edwin Paine, who were introduced in Sandman’s Season of Mists storyline and
later got their own 2001 comics miniseries. They also turned up in season 3 of Max’s Doom Patrol series.
</item 33>

<item 34>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
But he makes great use of expressive close-ups (often of himself) and shallow focus, with a few creepy It Follows-like shots of blurry figures approaching from
the distance, and a terrifically claustrophobic scene inside an MRI machine. Dalton’s college story, meanwhile, occasionally borders on campus-prank zaniness:
It includes what can only be described as a puke ghost, and there’s one amusing use of the horror movie cliche about the haunted little kid who makes terrifying
drawings of the ghouls only he can see. (Naturally, that kid grows up to become a star pupil in an insufferable freshman art class.)
</item 34>

<item 35>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
He’s blazing new trails and adding new layers to his style. Remarkably, getting the control that eluded him when he was younger has made his films more
consistently collaborative and creative, more fluid and alive than he was often allowed to be prior to the 2000s. He has the earned confidence of an artist who
has spent a lifetime in film and knows he’ll be able to find something great in post. His late period has only added to his legend. That should give Tarantino
cause to reconsider — alongside any artist or critic who thinks the creative spark has an expiration date.
</item 35>

<item 36>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Lanterns

The 2011 live-action Green Lantern movie (starring Ryan Reynolds as original Green Lantern Hal Jordan) was such a legendary flop it scuttled DC’s plans at the
time for a new interconnected superhero universe, and Reynolds keeps finding new ways to publicly mock the film, especially in his Deadpool movies. This
intergalactic-cop TV series will attempt to revive Hal Jordan for the screen, teaming him up with later Green Lantern John Stewart to “uncover a dark mystery.”

Paradise Lost

This series is set in Wonder Woman’s home island of Themyscira, and DC’s press release says it “focuses on the genesis and political intrigue of an island of
all women.” Sounds a bit like The Lord of the Rings: The Rings of Power, in terms of turning a familiar fantasy story into a political origin story. It’s
unclear as of yet whether Wonder Woman herself will have a role in the story, and whether the series will connect with Patty Jenkins’ Wonder Woman movies.
</item 36>

<item 37>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Aunt May has just gotten out of the hospital, and even noted jerk “Flash” Thompson has given Peter hell for how awful he’s been acting.

Peter, attempting to make things right, opts to cook the Thanksgiving meal all by himself, but mostly succeeds in ruining the kitchen. No worries — Gwen Stacy
and her father, along with a recuperating Aunt May and her doctor, all help out, and the episode concludes with a pleasant dinner. There are no big turns or
twists, aside from Aunt May revealing that she’s publishing a cookbook. It’s just simple, earned solace in a life marked by chaos. Peter even gets a kiss from
Gwen on his porch, a sequence conducted with a John Hughes sense of satisfying romantic flair.

Various Spider-Man comics have also dabbled in seeing what Thanksgiving looks like when you’re young and arachnid-themed, but real life has associated
Spider-Man with the holiday, too. Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade.
</item 37>

<item 38>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
The game also sheds its predecessor’s odious Spider-Cop bits in favor of establishing a deeper connection between the Spider-Mans and their city, giving the
game room to breathe while further immersing you in its heroes’ worlds and struggles.

And then there’s the gameplay. For my money, there are few experiences more enjoyable in gaming than swinging around New York City with Spider-Man. It was fun
in the early 2000s, and it’s still fun now. And with a bigger map to explore (and wings to fly, if you so choose to use them), there’s basically no limit to the
fun you can get up to. The deployment of two protagonists is seamless — Peter and Miles play differently, bringing their own stories and desires to the table —
and switching between them is effortless. It’s also the rare open-world game that doesn’t feel bloated with missions and side quests, instead leaving you
wanting more Spider adventures. —PV

7.
</item 38>

<item 39>
A timeline of Will Smith and Jada Pinkett Smith’s relationship amid separation (The Independent - Life and Style, 2023-10-12T18:10:45+00:00)
The altercation came after Rock said: “Jada, I love you. GI Jane 2, can’t wait to see you,” referring to Jada’s shaved head. While accepting the award for Best
Actor, Will became teary as said he was a “fierce defender of his family”.
</item 39>

<item 40>
Will Smith speaks to crowd about ‘tumultuous’ relationship with Jada Pinkett Smith (The Independent - Life and Style, 2023-10-19T16:16:20+00:00)
However, during an appearance on the Today show on 16 October, Jada clarified to Hoda Kotb that she and the Suicide Squad actor are “working hard” and are
“concentrating on healing” to get their relationship to a better place.

“There’s no finding another great love, and I think that’s the point,” she said. “It’s like we are in a place now that we are in a deep, healing space. And we
are really concentrating on healing the relationship between us.”

The Girls Trip star continued: “There’s no divorce on paper. We really have been working hard. That’s the whole thing. We are working very hard at bringing our
relationship together. Back to a life partnership.”
</item 40>

<item 41>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Outlaw Johnny Black

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Satirical Western

Run time: 2h 10m

Director: Michael Jai White

Cast: Michael Jai White, Anika Noni Rose, Erica Ash

Michael Jai White’s long-awaited follow-up to Black Dynamite is finally here: a “West-ploitation” movie about an outlaw who pretends to be a preacher and
settles in a new troubled town. The star and director spoke to us at length about the movie and the long road it took to get here.
</item 41>

<item 42>
NFL Fantasy Football Week 6 Lineup Decisions: Starts, Sits, Sleepers, Busts to know for every game (CBSSports.com, 2023-10-12T21:23:16+00:00)
It's a far cry from Week 1.

MINSHEW: Clearly more of a developed downfield thrower than Anthony Richardson and has thrown at Pittman on 26.4% of his routes run including four red-zone
targets and an end-zone target. Even last week when Minshew played less than a half Pittman saw a target once every five routes he ran with Minshew.

JAGUARS: Rank fifth with a 77.7% zone coverage rate, and they've consistently played a lot of zone in every game. They were also busted up by outside receivers
(over 17.0 yards per catch) last week against Buffalo and two weeks ago against the Texans. Pittman tends to play outside more than in the slot. Finally, an
outside receiver has scored at least once against the Jags in every game this season, including Pittman back in Week 1.
</item 42>

<item 43>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Sandman season 2

Netflix announced in November 2022 that its live-action series Sandman, based on Neil Gaiman’s seminal horror-fantasy comic, had been greenlit for a second
season. No date has been announced yet, and given the scale of the visual effects involved in the first season, it’s a fair bet that we won’t see the second
season until late 2024 or early 2025 at the earliest. Given that the first season covered the arcs of the first two Sandman graphic novels, Preludes and
Nocturnes and The Doll’s House (with side stories from the third volume, Dream Country), it’s a fair bet that season 2 will continue with book four, Season of
Mists, the heavily foreshadowed confrontation between Morpheus (Tom Sturridge) and Lucifer (Gwendoline Christie).
</item 43>

<item 44>
Can anyone survive Fortnite as a job? (Polygon, 2023-10-19T16:01:20+00:00)
A serious Twitch streamer who goes live daily or logs a full workday during their schedule will typically play much, much more than the people who are watching
them. Twitch streamers can’t stop; a mere two-day pause might mean losing thousands of paying subscribers. There’s a big incentive to keep up with live-service
updates as well. There’s some FOMO involved; while certain events with special prizes happen yearly, other big occasions, like season finales, only happen once.
But the numbers don’t lie: Your viewership will definitely be impacted by new patches that float the game in question to Twitch’s front page.

“Multiplayer games will start decreasing in viewership after the first week as the launch hype dies down and then it’s a question of how good the game is
perceived to be that will determine its long term fate,” says StreamElements’ public relations director Chase. “With those in for the long haul, viewership
tends to plateau and then spike when new content and patches are released or if it is supported by competitive events.”
</item 44>

<item 45>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Peter’s plot involves a retelling of the Venom origin story, with beats that will be recognizable to those even passingly familiar with the Spider-Man mythos.
If Spider-Man 2 is a game of retellings, Peter’s story clings most closely to the canon. If you’ve seen Sam Raimi’s Spider-Man 3, you’ll know the gist of what
to expect when the symbiote finds its way to Peter. It isn’t until the late game that Peter’s story starts pushing more strongly against our canonical memories
and into something more unique to Insomniac.

Miles’ story takes more liberties. Hailey Cooper, Miles’ love interest from his solo game, returns here, taking more of an MJ-esque role and carving out her own
identity separate from Miles’. American Sign Language is featured heavily throughout Miles’ missions because of Hailey’s presence, feeling as natural and
integral to the story as anything from the canon.
</item 45>

<item 46>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
There’s no comics-style backstory that demands pre-movie study.

Still, complete newcomers checking out the movie to see what the fuss is about may be confused for other reasons. The movie’s backstory is parceled out
awkwardly, in ways that are sometimes confusing. Characters’ decisions don’t always make sense. And there’s as much family melodrama as thrill-ride scares.
There isn’t much you can do about it except assure yourself that any confusion you experience in watching it isn’t just you. The movie is kind of weird, and not
because of some missing fan-only information that unlocks all its inconsistencies. The for-the-fans vibe Blum described applies more to whether the average
person will care about this story, not whether they can understand it.

Will Five Nights at Freddy’s experts enjoy the movie more?

Longtime fans hungering for an exploration or expansion of the game series’ continuing mythology should prepare themselves to see something more simplified.
</item 46>

<item 47>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feelings of loneliness spike during the holidays, meaning that there’s a good chance you’ll feel even more like Peter Parker when late November rolls
around.

It’s what makes the Thanksgiving scene in the 2002 Spider-Man film so engaging. That whole movie is an exercise in heart-on-your-sleeve sweetness. Everyone
who’s seen it knows the Thanksgiving dinner rapidly descends into chaos, with Norman Osborn ominously sticking his fingers into Aunt May’s sweet-potato
casserole, figuring out Spider-Man’s secret identity, and delivering a grossly sexist diatribe to his son Harry, leaving Harry and Mary Jane at odds with each
other. Aunt May and Peter clearly wind up with a ton of leftovers after everyone else storms out.

But for Peter, who’s just lost his Uncle Ben and has been facing the initial trials of being Spider-Man, there’s a nice moment of personal respite at the
opening of that scene, where he walks into a room where he knows he’s loved, bearing an offering of cranberry sauce,.
</item 47>

<item 48>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
You don’t get the full gale force of this film, its pleasures and its overarching point, without Scorsese reaching back to his past work, and relying on viewers
to recognize his voice in the storytelling.

Killers of the Flower Moon

Killers of the Flower Moon is perhaps most closely related to The Aviator as a film that appears to be a recognizable, industry-friendly form of awards-season
prestige. It’s also a return to Scorsese’s core themes and characters. It’s another cast of avarice-fueled, violent white American dumbasses gaining ill-gotten
wealth and making the country incrementally worse. It’s again about those men failing up, consolidating power, and shading the country more evil.

Killers adapts a major literary work of historical true crime from a New Yorker writer. The book delves into an obscure but monstrous historical incident that
is a cosmic truth in miniature, an evil that speaks to the heart of the entire dark American project.
</item 48>

<item 49>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I also do not know how Recreate made me completely OK with picking up these cute animals and flinging them into black holes, poison clouds, or freezing tundras.
The second the match starts in Party Animals, all those cute fluffy corgis, rabbits, kitties, and ducks become my enemies. I will beat them with a bat until
they can’t wake up anymore, and I won’t think twice about it. Party Animals may have made me a monster? I don’t know.

The Gang Beasts-esque wiggly physics mixed with the cute characters makes for a perfect party game of fluffy fighting. —Julia Lee

Related September games you might have missed

45.
</item 49>

<item 50>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
But in the end, the theme of the Spider-Verse movies is shaping up to be a story about people trying to be bigger and bolder themselves, trying to reach beyond
what they’re told they’re capable of, and do more. It’s no wonder that every part of Across the Spider-Verse is an attempt to outdo the first movie. The idea of
growing, of surpassing and ignoring everyone else’s limits, is the heart of this series’ heroes and their individual journeys. It looks like the movies
themselves are designed to follow suit.

Jawan

Where to watch: Available to stream on Netflix

Genre: Action thriller

Run time: 2h 50m

Director: Atlee

Cast: Shah Rukh Khan, Nayanthara, Vijay Sethupathi

The biggest Indian movie of the year has landed on Netflix.
</item 50>

<item 51>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
You play as both the yassified Peter Parker from Spider-Man Remastered and Miles Morales from his eponymous debut in 2020, switching between the two at your
leisure with all the speed afforded by the PlayStation 5’s SSD.

In fact, speed is one of the primary improvements on display in the first PS5-exclusive Spider-Man game. New to the series are web wings, which allow the
Spider-Mans to fly through the city, gliding over rooftops and gaining speed from wind tunnels without needing to rely on buildings, bridges, or other anchors
for web-slinging. (They’re especially useful when crossing the East River into the newly added Brooklyn and Queens.) Web-swinging is faster now as well,
enhancing that ineffable feeling of being Spider-Man as you glide through New York with ease. Whether you’re playing as Miles or Peter, traversing the three
available boroughs is an undeniable joy — so much so that I found myself using fast travel less frequently than in the previous two games, because movement felt
that much more fluid.
</item 51>

<item 52>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
Coming back to the ‘Wake Up’ mission, she’s seen sneaking past the hunters but she’s also capable of performing Takedowns on them. It wasn’t the case in the
previous title. Her missions have become better now with the devs actually spending time on giving her a chance to fight.

Coming back to the stun gun part, MJ is pretty skilled in using it, during missions like ‘Good Men’, thanks to her training with Sable. Also, now she’s got the
option to even run away and hide when spotted by enemies. The playability factor around her has increased a lot compared to being just an observer, for the most
part, in the first game. That’s the biggest and most visible difference.

Mary Jane and her missions in Spider-Man 2 allow the story to proceed without causing a problem for the central protagonists

The best thing about going through the quests showcasing Mary Jane is that they never try to overpower the ongoing story itself.
</item 52>

<item 53>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Happy Friday, Polygon readers! Each week, we round up the most notable releases to streaming and video rental, highlighting the biggest and best new movies for
you to watch at home.

This week’s biggest debut is Spider-Man: Across the Spider-Verse, which is now streaming on Netflix. That’s not all, as Insidious: The Red Door — the fifth
installment in the Insidious horror franchise — also arrives on the platform this week alongside Jawan, one of the biggest Indian action movies of the year.
There’s plenty more exciting releases this week too, with A Haunting in Venice now streaming on Hulu, the Italian superhero movie Freaks vs. the Reich on Prime
Video, plus the premiere of The Kill Room and Outlaw Johnny Black from director-star Michael Jai White on VOD.

Here’s everything new to watch this weekend!
</item 53>

<item 54>
Truth Behind Jake Paul Dating Dillon Danis’ Ex-girlfriend: Did MMA Star Break Up With Savannah Montano? (Essentially Sports, 2023-10-07T21:14:09+00:00)
ADVERTISEMENT Article continues below this ad

So, this tells the fans that Danis is no longer together with Montano. Anyway, the fans could look out for her attendance at Danis’s first-ever boxing debut on
October 14 at the AO Arena, Manchester. Although there were rumors about Danis pulling out, it seems the fight is still going to take place. So, her presence at
Danis’s ringside may make things even more complicated. What are your thoughts about this? Are they still together? Tell us in the comments section.

Watch This Story: Dillon Danis’s Relationship with His Wife
</item 54>

<item 55>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2024

Joker: Folie à Deux: Oct. 4

Todd Phillips’ acerbic 2019 drama Joker — effectively a DC-themed riff on Martin Scorsese’s King of Comedy — didn’t necessarily call for a sequel. It ends in a
wide-open place that invites viewers to write their own next chapter. But nothing breeds sequels like success, and the first Joker was a billion-dollar hit for
DC during a period where it was struggling to find that kind of payday from any of its properties. It feels like DC may have handed Phillips a blank check and a
free rein — at least, that seems like the best way to explain why the Joker sequel is reportedly a musical thriller with ​​Joaquin Phoenix reprising his role as
Arthur Fleck/the Joker, and Lady Gaga taking on the role of Harley Quinn. Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest,
Sophie.
</item 55>

<item 56>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
There’s so much going on in this patter-filled, joke-a-second comedy that it feels like 2023’s Everything Everywhere All at Once, right down to the big life
messages smuggled into all the goofery. It’s a high-speed joke-fest that doesn’t take Barbie any more seriously than she deserves — but does pay solemn homage
to all the ways, positive and negative, that Barbie fandom makes people feel. —TR

8. Spider-Man: Across the Spider-Verse

Directors: Joaquim Dos Santos, Kemp Powers, Justin K. Thompson

Cast: Shameik Moore, Hailee Steinfeld, Daniel Kaluuya

Where to watch: Netflix, or for digital rental/purchase

2018’s Spider-Man: Into the Spider-Verse changed the game for superhero movies and animated movies. It raised the bar for what a Spider-Man story could be, and
inspired animators to push the boundaries of what their movies could look like.
</item 56>

<item 57>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
Instead, it’s a fertile, vital body of work, virtuosic in entirely new ways, and as important in contextualizing his greatness as all the incredible work that
preceded it.

Old classics

Some of those nine movies are long-gestating bucket-list triumphs revisiting the pet subjects that defined Scorsese’s oeuvre in the 20th century, often directly
commenting on work from the front half of his career. 2002’s Gangs of New York, about how organized crime was woven into the operation of Scorsese’s native city
from its inception, is a prequel (or ancestor) to his classics Goodfellas and Casino. 2019’s The Irishman, about a man aging and dying with his guilt and sins
unresolved at the end of a life of crime, is intended as a bookend to his crime trilogy with Robert De Niro and Joe Pesci.
</item 57>

<item 58>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Whether it’s the upside-down kiss, the death of Uncle Ben or Jefferson Davis, or simply the spider bite that starts it all, Spider-Man 2099 believes that
upholding these canon events, even the painful ones, is key to preserving the stability of each strand of reality. Miles disagrees, believing it is Spider-Man’s
job to resist the seemingly inevitable — to always do what’s right, not simply what’s preordained. The film ends with a promise of pushing back against
Spider-Man 2099, and with it, the established canon.

I was hungry for Spider-Man 2 to push back in a similar way — to treat the established story like Angela Carter repurposing “Little Red Riding Hood” to her
whims, like Michael Cunningham plumbing “Rumpelstiltskin” for a story about the desire to have a child, or like Helen Oyeyemi and her body of work.
</item 58>

<item 59>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
In an uncharitable light, it plays as performative, as an older relative getting worked up over a police brutality Facebook post. It’s too early to decide how
this film tags into our present historical moment, or how it will be remembered. We all have many viewings of and conversations about Killers of the Flower Moon
ahead of us.

A true master

This century has seen Scorsese zigging and zagging as always, but because he’s following his own varied interests, not because he’s moving from compromise to
compromise, as he once had to in order to continue working. An artist who spent the first 30 years of his career desperate for autonomy finally got it, and now
he’s taking full advantage: telling difficult stories, setting his own mega-budgets, and demanding every minute of run time he feels each picture demands.

He’s used this autonomy to revisit his work and revise the record, creating great new films as well as adding crucial addendums to past work. He’s blazing new
trails and adding new layers to his style.
</item 59>

<item 60>
5 things to know before watching the Five Nights at Freddy’s movie (Polygon, 2023-10-26T21:11:01+00:00)
Can you enjoy Five Nights at Freddy’s if you haven’t played the games?

At a Blumhouse panel for the recent New York Comic Con, producer Jason Blum explained that he thinks a lot of adaptations — books, comics, and games alike —
falter when creators try to simultaneously please hardcore fans and remain accessible to new audiences. According to Blum, his company finally cracked Five
Nights at Freddy’s by deciding not to make it with new audiences in mind. In other words, they ultimately decided to just please the fans.

It was a clear applause-bait line at a convention filled to the brim with, yes, fans. But that doesn’t mean the movie is hard to follow. It starts from scratch,
mimicking the premise of the first game: A guy named Mike (Josh Hutcherson) is hired as overnight security guard for an abandoned Chuck E. Cheese-style arcade
restaurant, and encounters some animatronics with strange and eerie origins. There’s no comics-style backstory that demands pre-movie study.
</item 60>

<item 61>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
The Kill Room

Where to watch: Available to rent on Amazon, Apple, and Vudu

Genre: Dark comedy thriller

Run time: 1h 38m

Director: Nicol Paone

Cast: Joe Manganiello, Samuel L. Jackson, Uma Thurman

Joe Manganiello stars as a hitman turned artist in this funny comedy about how the worlds of fine art and high crime aren’t so separated after all. When he
turns to art as a method of laundering money, the hitman becomes an unexpected overnight sensation in the high-art scene.
</item 61>

<item 62>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Guardians of the Galaxy and Suicide Squad director James Gunn and producer Peter Safran are taking the reins of the newly branded DC Studios, and they’re
bringing a long-term plan to DC with them. The new projects on the DC docket are meant to reboot or reshape DC’s best-known heroes, while reaching into some
deeper corners of the DC canon for more obscure characters. And it’s wrapping up the DCEU era, replacing it with just “the DCU,” a mainline continuity that will
exist alongside other DC film and TV universes. Here’s everything DC is currently planning to put in theaters and on TV screens, in 2023 and well into the
future.

[Ed. note: This story only covers Warner Bros.’ DC projects and doesn’t include The CW’s Arrowverse, which is effectively ending in 2023.]

2023

Aquaman and The Lost Kingdom: Dec.
</item 62>

<item 63>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
This is a story, broken into pieces, we learn. One that ended badly before. One that will likely end badly again. Will you still put it together?

Alan Wake 2 is full of words but few are in order. It is set in places of confused geography, full of people who don’t quite belong, in rooms that might be
different each time you enter them. Alan Wake 2 is wrong, and you must make it right, if you can.

Remedy Entertainment’s assured masterpiece is less about stories and more about dreams — the way they can slip from frightening to absurd at a moment’s notice,
about how we can get lost in them, learning from our experiences or succumbing to insecurities. How video games can mimic their shape, or lack thereof.

It is about repetition, and the way we find meaning in stories and people and places by returning to them over and over, wondering if we changed or they did.
</item 63>

<item 64>
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
</item 64>

<item 65>
The best surprise vampires in things you did not expect to see vampires (Polygon, 2023-11-15T18:02:00+00:00)
Preacher: Cassidy

Preacher, the comic book by Garth Ennis and Steve Dillon — and its subsequent TV adaptation from Sam Catlin, Evan Goldberg, and Seth Rogen — starts with the
basic assumption that God is real, and has walked out on the job. A lot of things are fair game at that point, namely angels, demons, and the like, but it’s
still a shock when you meet Cassidy. A brawling, hard-drinking Irishman that’s already a load of fun when you meet him, Cassidy shoots up the power rankings
when he sinks his teeth into a poor sap, because he’s a vampire.

Spider-Man: The Animated Series: Morbius

Hands down the ballsiest Spidey villain to adapt in a cartoon for children that censors wouldn’t even let use the words “sinister” or “kill,” Morbius — the
Living Vampire — frankly had no business being in a Saturday morning lineup.
</item 65>

<item 66>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Beleaguered movie theaters got a boost this year from film fans who made an event out of seeing two radically different movies back-to-back: a slyly satirical
movie about a living fashion toy, and a heavy, grim historical drama about the father of the atomic bomb. While Barbie and Oppenheimer do have a clear theme in
common, they’re still radically different movies. But the way they collectively dominated the 2023 box office and the cultural discussion is a strong reminder
that regardless of tone or topic, a great story, told passionately, stands out. Cinephiles can be drawn to just about any movie in any genre, if it’s well
crafted and engaging, no matter what kind of fruit it is.

Here at Polygon, we have pretty eclectic collective tastes: Some of us are hungriest for action, horror, epic fantasy, or challenging science fiction, while
some of us prefer complicated drama or dark neo-noir, and others are drawn to musicals, comedy, animated adventures, and other lighter fare.

Our best-of-the-year list reflects that range.
</item 66>

<item 67>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Here’s everything new to watch this weekend!

New on Netflix

Spider-Man: Across the Spider-Verse

Where to watch: Available to stream on Netflix

Genre: Superhero action

Run time: 2h 20m

Director: Joaquim Dos Santos, Kemp Powers, Justin K. Thompson

Cast: Shameik Moore, Hailee Steinfeld, Oscar Isaac

The highly anticipated follow-up to 2018’s Spider-Man: Into the Spider-Verse sees Miles Morales facing off not only against a dimension-hopping nemesis in the
form of the Spot, but a whole multiverse of Spider-Mans, Spider-People, and even a Spider-Dinosaur as he attempts to save the day once again.

From our multiversal review:

Not every theme and plot and moment in Across the Spider-Verse lands, particularly with the other part of this story still most of a year away.
</item 67>

<item 68>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Zazie Beetz will be back as Arthur’s neighbor and entirely unwitting love interest, Sophie.

Creature Commandos

James Gunn is listed as the writer for this seven-episode adult animated series, which will also include Suicide Squad head Amanda Waller. This time, she
“creates a black ops team out of monstrous prisoners.” The team will include Frankenstein’s monster (here called Eric Frankenstein and voiced by David Harbour),
who has a long, complicated DC history of his own. The series is set to star Frank Grillo (Prison Break) as Rick Flag Sr., Maria Bakalova (Guardians of the
Galaxy Vol. 3) as Princess Ilana Rostovic, Indira Varma (Game of Thrones) as the Bride of Frankenstein, and Alan Tudyk (Harley Quinn) as Doctor Phosphorus.
</item 68>

<item 69>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Marvel built its comic book revolution on the back of one idea: rendering colorful superheroes as relatable. In the mid 20th century, DC do-gooders were
essentially square-jawed Sunday school teachers, which let Marvel corner the market on heroes with human depth and fragility. The Fantastic Four were vulnerable
to insecurity and in-fighting. The X-Men represented the cost of bigotry on a wide scale.

But Spider-Man represented this new wave best. Consumed by youthful everyman angst and desperate to find balance in his life, Spider-Man is broadly sympathetic.
We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving. And Spider-Man
writers know it.

Thanksgiving has a complicated history, tied to the roots of American colonialism, and more recently processed through a lens of joyful capitalism. But feeding
friends and family is perhaps one of the most humane acts we can pursue.
</item 69>

<item 70>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
That film essentially birthed an entire genre of filmic language, but The Departed goes a step beyond the original. It was constantly being rewritten and
adjusted with the actors on the fly. The result was exhilarating seat-of-the-pants filmmaking that Scorsese and his longtime creative partner, editor Thelma
Schoonmaker, wrangled into a coherent film.

Scorsese brought this cokey, maximalist energy to The Wolf of Wall Street, a pointed comedy as wildly, masterfully off the rails and improvisational as anything
he’s ever done. (Jonah Hill perhaps smuggled in a touch of Judd Apatow’s “Keep the camera rolling” ethos.) You can feel a healthy dose of Scorsese’s
established-filmmaker confidence (and a touch of “Eh, why not? Fuck it!” thinking) as DiCaprio’s Ferrari changes color mid-narration during a highway blowjob,
and when DiCaprio addresses the camera directly, urging the audience not to worry about specific plot points.
</item 70>

<item 71>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
With the recent release of Marvel’s Spider-Man 2, Insomniac has witnessed a huge appreciation from fans for delivering a memorable sequel. Not to mention the
epic portrayal of Peter Parker and Miles Morales fighting against some of the strongest villains. But then, there’s MJ as well who has made a comeback from the
first installment. And it’s her arc that has actually received an impressive elevation this time. Especially when considering the poorly received missions
featuring Mary Jane in the original game from 2018.

America’s Favorite Video Today

So what’s actually changed in 2023 that has created a vast difference? There are certain key reasons making Mary Jane in Spider-Man 2 a treat to watch. Not just
from the perspective of her relationship with Peter, but her own missions showcasing a different side this time. One can even call her the third unofficial
superhero in the latest PS5 exclusive.
</item 71>

<item 72>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Or, to give a more mass media example, like Star Wars: Visions, which gives creative leeway to a variety of animators and storytellers to play with the very
basics of what makes Star Wars Star Wars.

Spider-Man 2 doesn’t go quite this far, but where it does go is still satisfyingly slant. Wrinkles to critical characters and plot points will keep even
seasoned Spider-fans on their toes. Moreover, it plays like a dream, with smart additions to combat that enhance the power fantasy of being Spider-Man. Each
Spider-Man now has a full arsenal of special moves and gadgets that complement one another, making even the most basic combat encounter feel like a
choreographed ballet of kicks, thwips, and zaps. The game is beautifully rendered, taking full advantage of the newest PlayStation, with such detailed textures
on each unlockable suit that you’d swear they were real, and ray-traced window after ray-traced window in a loving rendition of New York.
</item 72>

<item 73>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
Waller

Gunn’s hilarious, heartfelt Suicide Squad spinoff series Peacemaker was greenlit for a second season in February 2022, but Gunn has now said he’s going to have
to put that project on hold for a bit to concentrate on other things — including Waller, the previously untitled Peacemaker spinoff centered on Viola Davis as
no-nonsense supervillain resource broker Amanda Waller. The character, introduced to DC’s movie continuity in David Ayer’s little-loved 2016 version of Suicide
Squad (though she’s been part of the comics continuity since 1986), is one of the few elements of Ayer’s movie that was held over in Gunn’s 2021
reboot/reshuffle movie, called The Suicide Squad. She also turned up for a quick cameo in Black Adam. Now she’s getting her own show — which “features Team
Peacemaker,” so it may feel like a stealth season 2, depending on who from the team winds up on the series.
</item 73>

<item 74>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
But how?

Compared to the lack of interactivity in 2018’s Spider-Man, MJ is now capable of tackling the enemies around her. In the first title, all she could do was to
move from one point to the other. Now, she’s able to do more than that – make pretty good and lethal moves. To be precise, the devs have given MJ a little
firepower, if not in too much capacity. Now, she’s got weapons instead of being clueless in moments of being vulnerable against armed enemies. Yes, she hasn’t
got any superpowers, but she’s got other means to help herself out.

Due to such involvement, players have now got a reason to be more invested into a given mission focusing on MJ. It wasn’t the case at all taking into
consideration her previous iteration in Insomniac’s original game. Mary Jane can now handle the circumstances in her own manner, thus breaking free from the
linear approach of just trying to hide and escape.
</item 74>

<item 75>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feeding friends and family is perhaps one of the most humane acts we can pursue. It’s less about the spirit of giving, and more about admitting that people
have innate, basic needs, like food and social comfort, and that those needs are best fulfilled when people work in tandem. Honestly, we should try to do it
more on every other day of the year: The number of people volunteering to help feed the homeless and families in need peaks around November and December, but
that energy needs to be carried through the preceding 10 months, too.

Spider-Man represents these needs, even though very few of us have to balance photojournalist work, a fraught dating life, and pummeling Dr. Octopus. Peter
Parker is often the most financially strapped among his Avengers buddies, and frequently the loneliest, too. Those are recognizable traits among many young
people, even those without radioactive-spider powers. When you’re growing up and trying to figure out the world, it’s easy to feel lost and isolated.
</item 75>

<item 76>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Yep — along with everything else, Poor Things is also goth Barbie.

Adapted from a cult novel by the eccentric Scottish writer, illustrator, and typesetter Alasdair Gray, Poor Things reworks Frankenstein to explore themes of
feminism, sex, and social rot. Lanthimos doubles down on the baroque stylings of The Favourite to create an astonishingly visually dense film composed of
outlandish costumes, lavish sets, and fantastical painted backgrounds, often shot through woozy fish-eye lenses. It’s also, surprisingly, one of the year’s most
hilarious movies, in no small part thanks to Mark Ruffalo’s game campery and McNamara’s cleverly twisted syntax. But the movie is Stone’s. Her performance is
fearless, frank, funny, and intensely physical.
</item 76>

<item 77>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
The eight-episode series, produced by Reeves and overseen by Agents of SHIELD writer-producer Lauren LeFranc, is meant as a lead-in for The Batman Part II, so
even though it’s currently undated, expect it before the movie hits.

Arkham Asylum

The actual title for the second of Reeves’ two HBO Max The Batman spinoffs hasn’t been revealed yet, but this one is meant to be a prequel to the movie.
Previously planned as a series about the workings of the Gotham police department, the series mutated in the development stage, and Boardwalk Empire creator
Terence Winter, originally in charge of the series, departed the project. Reeves has described this series as an origin story for characters seen in Arkham in
The Batman, with more of a horror movie or “haunted house” feel. The Staircase’s Antonio Campos is now the showrunner for the series.
</item 77>

<item 78>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
As Gracie and Elizabeth obsess over little details, May December lets Melton make the case for the quiet, seething corruption at the heart of the film. —ZM

Top 10

10. Showing Up

Director: Kelly Reichardt

Cast: Michelle Williams, Hong Chau, John Magaro

Where to watch: Digital rental/purchase

Lizzy (Michelle Williams) is a 30-something artist living on campus at her hometown art school. She wants two things in life: time to prepare for her
exhibition, and her landlord, Jo (Hong Chau), to fix her hot water heater. Superficially, there’s little more to Lizzy’s plight, for the better. Director and
co-writer Kelly Reichardt never permits the stakes to rise too high or the plot to get too complicated. Because this film, at its core, is about working artists
working in less financially stable mediums, like ceramic figurines and trippy visual projections.
</item 78>

<item 79>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
The DC Extended Universe, as DC Comics’ movie and TV line was originally called, has gone through some major growing pains since it launched with Zack Snyder’s
Man of Steel in 2013. Looking for ways to compete with Marvel Comics’ massively popular Marvel Cinematic Universe, DC has repeatedly switched tracks and tactics
on the DCEU, sometimes steering toward a single interconnected universe, then backtracking into a more director-driven plan where movies like Todd Phillips’
Joker could be made without fitting into any specific previous continuity.

In 2023, the Snyderverse is coming to an end, slowly rolling to a halt with movies like Shazam! Fury of the Gods, The Flash, and Blue Beetle, all leading up to
the grand finale of Aquaman and the Lost Kingdom. Once the last bits of the DCEU make their way to theaters, it will be time to usher in a new era of DC movies.
</item 79>

<item 80>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
One of the Miles-specific side quests involves learning about the artistic history of Harlem, and it entwines the real-life histories of Josephine Baker,
Langston Hughes, and Charlie Parker with the fictional history of SHIELD and other Marvel properties. An early mission shows off Insomniac’s technical chops
with a section that cleverly borrows a trick from Ratchet & Clank: Rift Apart. Of the two Spider-Mans, it’s Miles who feels more alive with the possibility to
break free of expectation and canon.

[Ed. note: Spoilers follow for Spider-Man: Across the Spider-Verse.]

The recent Spider-Man: Across the Spider-Verse has an obsession with “canon events,” pivotal moments in a Spider-Man’s life that the film’s villain, Spider-Man
2099, is dedicated to maintaining across the multiverse.
</item 80>

<item 81>
The best surprise vampires in things you did not expect to see vampires (Polygon, 2023-11-15T18:02:00+00:00)
The clever folks behind Spider-Man got around the strict content standards of their network with a loophole that might be even creepier than how vampires
normally feed: by giving him little suckers on the palms of his hands that drain victims of “plasma.”

Hilariously, a creative application of vampirism would be how the show would later adapt another villain normally too violent for kids TV: Carnage, a serial
killing symbiote who now “drained” his victims’ “life force” instead of cutting them to bits.

Spider-Man: Across the Spider-Verse: Miguel O’Hara

Miguel O’Hara, the Spider-Man of 2099, is one of Spider-Man: Across the Spider-Verse’s big enigmas. Comics readers will know plenty about his whole deal, but
the movie doesn’t really explain much — except for a brief scene where he’s shown to have fangs, and an offhand reference that he is kind of a vampire? Sure,
he’s a jerk, but he’s still kind of cool.
</item 81>

<item 82>
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
</item 82>

<item 83>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
Next stop: the Bronx.

Marvel’s Spider-Man 2 will be released Oct. 20 on PlayStation 5. The game was reviewed using a pre-release download code provided by Sony Interactive
Entertainment. Vox Media has affiliate partnerships. These do not influence editorial content, though Vox Media may earn commissions for products purchased via
affiliate links. You can find additional information about Polygon’s ethics policy here .
</item 83>

<item 84>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
2025

Superman: Legacy: July 11

The first of the new Gunn/Safran projects to actually get a release date, Superman: Legacy seems expressly designed to break the DCEU mold. Henry Cavill is
moving on from the Superman role, and Gunn, who plans to script the film himself, made several pointed references to this film being a break from past visions
of Superman: He called the film “the true foundation of [their] creative vision for the DC Universe,” and said he “can’t wait to introduce [their] version of
Superman.” The advance description for this one currently just says it’ll feature the character “balancing his Kryptonian heritage with his human upbringing.”

The Batman Part II: Oct. 3

The long, contentious history behind 2022’s The Batman didn’t leave much room for hope: As one of those movies that got bounced from team to team for years,
changing shape at every bounce, it looked like a project that was destined to be driven more by market calculation than by a specific vision.
</item 84>

<item 85>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
A Haunting in Venice is his best yet, as Branagh’s confidence as director and performer in this mode only continues to grow. It’s perfect fall viewing.

Quiz Lady

Where to watch: Available to stream on Hulu

Genre: Comedy

Run time: 1h 39m

Director: Jessica Yu

Cast: Awkwafina, Sandra Oh, Will Ferrell

Sandra Oh and Awkwafina play a pair of estranged sisters who try to win big on a game show to pay off their mom’s debts. The supporting cast includes Will
Ferrell, Jason Schwartzman, Tony Hale, and the late Paul Reubens.

New on Prime Video

Freaks vs.
</item 85>

<item 86>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
“Those slavering jaws; the lolling tongue; the rime of saliva on the grizzled chops.”

This is Angela Carter, not on Venom, but on the wolf from “Little Red Riding Hood” in The Bloody Chamber, her collection of reimagined folk and fairy tales.
It’s actually from “The Company of Wolves,” which is one of three retellings of “Little Red Riding Hood” in the collection, each with its own spin on the
familiar tale.

Polygon Recommends is our way of endorsing our favorite games, movies, TV shows, comics, tabletop books, and entertainment experiences. When we award the
Polygon Recommends badge, it’s because we believe the recipient is uniquely thought-provoking, entertaining, inventive, or fun — and worth fitting into your
schedule. If you want curated lists of our favorite media, check out What to Play and What to Watch .
</item 86>

<item 87>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Sure, everyone in that room is trying to hold things together. Mary Jane wants to impress Norman, Harry wants to impress Norman and Mary Jane, Peter loves Mary
Jane but doesn’t dare hurt Harry, and Aunt May is finishing what’s presumably a dope turkey, in an attempt to care for all of them.

It’s the trying that counts: Life can be hard, weird, and cruel, but while sitting down for a meal with our nearest and dearest, maybe for at least a little
while, we won’t need to have it all figured out.

Not all Spider-Man Thanksgiving dinners end up falling apart, and some succeed in reminding Peter that he isn’t alone. The first season of The Spectacular
Spider-Man ends with a bang: One of Peter’s closest pals, Eddie Brock, has become Venom, and has threatened everyone Peter holds dear. This comes just after
Peter himself was infected by the symbiote, and went through the now-iconic throes of pushing everyone away.
</item 87>

<item 88>
Spider-Man 2 toys with Marvel history to be as much fun as possible (Polygon, 2023-10-16T14:00:00+00:00)
As for our heroes, Miles and Peter return from their individual outings for a dramatic team-up. Their two stories are threaded together, allowing for randomized
encounters where Peter might show up while you’re playing as Miles (or vice versa), with the two of you tag-teaming a random criminal and slamming them into the
ground in what feels like the world’s least fair fight. And yes, they do the Spider-Man-pointing-at-Spider-Man meme in-game. And yes, it is charming the first
time (though maybe less so the second). The main story sometimes swaps Spider-Mans mid-mission, as in the dramatic opening sequence where Sandman coats the city
in dust as Peter and Miles rush to help firefighters and first responders, with eerie echoes of 9/11. Side missions often allow either Spider-Man to tackle
them, but the main story beats require one or the other, as each hero is allowed his full particularity as a singular Spider-Man.
</item 88>

<item 89>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
They also turned up in season 3 of Max’s Doom Patrol series. The initial press release on Dead Boy Detectives said it would have an eight-episode first season,
described as “a fresh take on a ghost story that explores loss, grief, and death through the lens of Edwin Payne (George Rexstrew) and Charles Rowland (Jayden
Revri), two dead British teenagers, and their very alive friend, Crystal Palace (Kassius Nelson).”

Peacemaker season 2

Alongside everything else in this universe with his name on it, Gunn hasn’t forgotten Peacemaker. According to the director, Christopher Smith will be making
his return for season 2, sometime after Gunn wraps production on Superman: Legacy. This likely means that season 2 won’t be coming along until sometime in 2025
or even later, but hey, at least we know it’s out there.
</item 89>

<item 90>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
And if you just enjoy a good yarn and have access to basically any screen and $15, then you’re a perfect mark too. It runs as well on console and PC as it does
on iOS and Android, so don’t fret about where you play, just do so and soon! Before Square Enix stops investing in all these oddities. —Chris Plante

22. Marvel’s Midnight Suns

Developer: Firaxis Games

Where to play: Nintendo Switch, PlayStation 4, PlayStation 5, Windows PC, Xbox One, and Xbox Series X

[Ed. note: Marvel’s Midnight Suns was released in 2022, but it just barely missed the cutoff for our best video games of 2022 list, so it’s eligible for our
2023 awards.]

I know what you’re thinking: Another licensed Marvel game? Come on, right? But hear me out. I played Marvel’s Avengers, too, and this isn’t that.
</item 90>

<item 91>
Martin Scorsese’s late-stage career has proved Quentin Tarantino wrong (Polygon, 2023-10-26T14:02:00+00:00)
On screen, it’s Scorsese’s first true Western, filled with gorgeous vistas and Oklahoma horizons that would bring a tear to the eye of John Ford, one of
Scorsese’s many idols.

But the film is much stranger than it sounds on paper. Scorsese’s take on the material largely operates outside the text, centering the story on a peripheral
character we know little about from David Grann’s book. Miraculously, the film also has its funny elements, incorporating The Departed and The Wolf of Wall
Street’s nervy energy, featuring long and zany improvisational exchanges on grim subject matter, sudden wild swings in tone, and one major decision that should
spark more debate and differences of opinion than any narrative device Scorsese has ever employed. It’s a suspenseful, moving film with many brilliant parts.

Killers is a classic Scorsese Catholic reckoning.
</item 91>

<item 92>
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
</item 92>

<item 93>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
New on Prime Video

Freaks vs. the Reich

Where to watch: Available to stream on Prime Video

Genre: Superhero/circus war movie

Run time: 2h 21m

Director: Gabriele Mainetti

Cast: Claudio Santamaria, Aurora Giovinazzo, Pietro Castellitto

This offbeat Italian superhero movie follows a group of circus performers in World War II who are sought after by the Nazis and team up to stop them. I have
heard it’s funny, sweet, and has strong action — definitely on my weekend watchlist.
</item 93>

<item 94>
The best movies of 2023 so far (Polygon, 2023-10-07T12:38:36+00:00)
Without plot filling the film, we have time to just be present in this beautiful Pacific Northwest scene.

Reichardt’s genius is getting the audience giggling at the artists but never the art. For example, it’s funny to think that an artist dedicated a year of her
life to crocheting a jumpsuit. Except then, in Showing Up, you see the outfit and it’s beautiful — an intentional undermining of the punchline. A teacher smugly
opines on ceramics, but each piece he holds up is so lovingly crafted that they confidently speak for themselves.

This decision (rib artists, celebrate art) sets the tone. We humans are artifice, a bunch of contradictory masks that we put on to match the situation and the
crowd. But our creations — when we commit to a craft, whatever medium it may be — are an expression of our most vulnerable selves.
</item 94>

<item 95>
Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way (Essentially Sports, 2023-11-01T21:00:28+00:00)
The game’s narrative remains intact, no matter what. There never occurs a feeling of the devs trying too hard to make it seem like her missions are stretching
out of their boundaries.

Although they are designed well this time but one has to commend Insomniac’s team to keep a limit on not making MJ’s gameplay moments seem forced or going
beyond a certain point. They serve their purpose without affecting the main plot of Marvel’s Spider-Man 2 or overlapping with Peter or Miles’ stories. In a way,
each character has been given a good amount of time to shine. At times, it may feel like MJ’s playable sequences should have been extended. But it’s nice to see
that the devs decided to not do that; sometimes less is more and fine.

Despite Spider-Man 2 being more of a Peter’s story (in its entirety), MJ’s role has been given equal significance too. There’s a visible growth in her arc
especially considering her profession as a journalist.
</item 95>

<item 96>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade. Other Marvel characters have appeared on floats, but only
Spider-Man has been inflated to 80 feet and pulled through the Upper West Side. (That also makes him the only Marvel character to have his head horrifically
torn open by a tree branch.)

From his first comic book appearance, Spider-Man has been a reminder that life is hard and complicated, and being a superhero doesn’t preclude anyone from
experiencing ordinary frustrations, setbacks, and confusion. But no matter how Spider-Man’s Thanksgiving escapades turn out, they remind readers and viewers
that the holiday is about the hope of mutual connection, support, and nurturing. Even if the girl of your dreams is out of reach and the Green Goblin is on your
case, a table, some friends, and Aunt May’s rad turkey might just make everything better for a little while.
</item 96>

<item 97>
The future of DC movies in 2022 and beyond (Polygon, 2023-10-24T15:00:36+00:00)
This will be Damian’s first appearance in a live-action film, though the character has appeared in animated form in TV and movies. The continuity of this film
will be separate from Matt Reeves’ planned The Batman trilogy; it’s meant to establish the new DCU continuity for Batman and Robin.

The Authority

The WildStorm imprint — founded as an independent company in 1992 and bought up by DC in 1999 — maintained its own superhero universe separate from DC’s, until
DC shut it down in preparation for its characters to join the mainline DC Universe in the company’s 2011 reboot. In its day, Wildstorm published independent and
creator-owned comics, alongside superhero series like Warren Ellis and Bryan Hitch’s The Authority, about a group of seven heroes with high-concept superpowers,
like the Engineer, a woman with nanotech blood, and Jack Hawksmoor, a man who psychically bonds to cities. This will be the first appearance of WildStorm canon
in live-action film.
</item 97>

<item 98>
Spider-Man: Across the Spider-Verse on Netflix, A Haunting in Venice, and every new movie to watch this weekend (Polygon, 2023-11-03T22:15:00+00:00)
Sly

Where to watch: Available to stream on Netflix

Genre: Documentary

Run time: 1h 35m

Director: Thom Zimny

Cast: Sylvester Stallone, Arnold Schwarzenegger, Quentin Tarantino

This documentary takes a close look at the life of one of the great American movie stars and film writers: Sylvester Stallone.

Nyad

Where to watch: Available to stream on Netflix

Genre: Biographical sports drama

Run time: 2h 1m

Directors: Jimmy Chin, Elizabeth Chai Vasarhelyi

Cast: Jodie Foster, Annette Bening, Rhys Ifans

Nyad tells the (questionably) true story of swimmer Diana Nyad, who swam from Cuba to Florida at 64 years old, among many other swimming accomplishments.
</item 98>

<item 99>
The best video games of the year so far (Polygon, 2023-10-31T16:30:45+00:00)
I played Marvel’s Avengers, too, and this isn’t that. It might seem like it’s going to be at first, because Midnight Suns makes the grave error of introducing
Iron Man and Doctor Strange as its tutorial characters, and these two might just be the most irritating characters in the entire video game. (I have beaten the
game, so I am allowed to make this call.) You must press on and give Midnight Suns time to win you over. Because it has so, so much more to offer than it may
appear in its first few hours.

Picture the romance and humor of Fire Emblem: Three Houses, combined with the high-stakes tactical battles of XCOM 2 — that’s what Midnight Suns becomes in its
mid-game and endgame. It’s a card-based strategy game, and each hero has their own customizable deck. I started off favoring Captain Marvel, Magik, and Blade,
simply because their moves and hilarious dialogue kept me entertained, but I soon realized that every single character has something exciting or unexpected to
bring to the battlefront.
</item 99>

<item 100>
The best tabletop RPGs we played in 2023 (Polygon, 2023-12-13T15:00:00+00:00)
The game is utterly devoted to giving the players the feeling of playing a hero in the Marvel universe with tons of cool powers and power categories based on
key Marvel heroes like Spider-Man and the Hulk. The system is easy, streamlined, and really carries the four-color comic book flavor. Probably one of the
coolest aspects of the game is that while there is an exhaustive list of Marvel heroes and villains at the back of the book, you can create your own hero with
their own power set and use that “bestiary” of heroic and villainous figures as allies and enemies in your own heroic adventures. For those of us who’ve always
wanted to make their own X-Gene Mutant, Spirit of Vengeance, or Spidersona, this is a must-have book that will bring hours of enjoyment.

Masks of the Masks

Superhero comics, for all their many flaws, remain my life’s great passion.
</item 100>
=== END ITEM id=0006d5b60d468a14 ===
